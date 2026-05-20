const { GoogleGenAI, Modality } = require('@google/genai');
const { BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const { saveDebugAudio } = require('../audioUtils');
const { getSystemPrompt } = require('./prompts');
const {
    getAvailableModel,
    incrementLimitCount,
    getApiKey,
    getOpenrouterApiKey,
    getAnthropicApiKey,
    getPreferences,
} = require('../storage');
const { connectCloud, sendCloudAudio, sendCloudText, sendCloudImage, closeCloud, isCloudActive, setOnTurnComplete } = require('./cloud');
const { PROVIDERS, sendToProvider, CLOUD_TIMEOUT_MS, buildFailoverChain } = require('./providers');

// Provider mode: 'byok' or 'cloud'
let currentProviderMode = 'byok';

// Conversation history for text provider context
let conversationCtxHistory = [];

// Conversation tracking variables
let currentSessionId = null;
let currentTranscription = '';
let conversationHistory = [];
let screenAnalysisHistory = [];
let currentProfile = null;
let currentCustomPrompt = null;
let isInitializingSession = false;
let currentSystemPrompt = null;

function formatSpeakerResults(results) {
    let text = '';
    for (const result of results) {
        if (result.transcript && result.speakerId) {
            const speakerLabel = result.speakerId === 1 ? 'Interviewer' : 'Candidate';
            text += `[${speakerLabel}]: ${result.transcript}\n`;
        }
    }
    return text;
}

module.exports.formatSpeakerResults = formatSpeakerResults;

// Live API model fallback: track if primary model's quota is exhausted
let liveModelQuotaExhausted = false;

// Audio capture variables
let systemAudioProc = null;
let messageBuffer = '';

// Debounce: dispatch transcription after silence (no new VALID chunks)
// This fires BEFORE generationComplete in most cases, saving 1-3 seconds
const TRANSCRIPTION_DEBOUNCE_MS = 2000; // 2s silence = speaker actually finished
let transcriptionDebounceTimer = null;
let isCapturingSpeech = false; // tracks if we've shown "Capturing..." status

/**
 * Quick check: does this chunk contain any Latin characters (likely English)?
 * Non-Latin noise should NOT reset the debounce timer.
 */
function chunkHasLatinChars(text) {
    return /[a-zA-Z]/.test(text);
}

function resetTranscriptionDebounce() {
    clearTranscriptionDebounce();

    // Show "Capturing speech..." on first chunk so user knows we're hearing something
    if (!isCapturingSpeech && !isDispatching) {
        isCapturingSpeech = true;
        sendToRenderer('update-status', 'Capturing speech...');
    }

    transcriptionDebounceTimer = setTimeout(() => {
        isCapturingSpeech = false;
        if (currentTranscription.trim() !== '') {
            console.log('[DEBOUNCE] Silence detected, dispatching...');
            sendToRenderer('update-status', 'Analyzing...');
            const text = currentTranscription;
            currentTranscription = '';
            messageBuffer = '';
            dispatchToTextProvider(text);
        } else {
            sendToRenderer('update-status', 'Listening...');
        }
    }, TRANSCRIPTION_DEBOUNCE_MS);
}

function clearTranscriptionDebounce() {
    if (transcriptionDebounceTimer) {
        clearTimeout(transcriptionDebounceTimer);
        transcriptionDebounceTimer = null;
    }
}

// Reconnection variables
let isUserClosing = false;
let sessionParams = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY = 2000;

function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        windows[0].webContents.send(channel, data);
    }
}

// Build context message for session restoration
function buildContextMessage() {
    const lastTurns = conversationHistory.slice(-20);
    const validTurns = lastTurns.filter(turn => turn.transcription?.trim() && turn.ai_response?.trim());

    if (validTurns.length === 0) return null;

    const contextLines = validTurns.map(turn => `[Interviewer]: ${turn.transcription.trim()}\n[Your answer]: ${turn.ai_response.trim()}`);

    return `Session reconnected. Here's the conversation so far:\n\n${contextLines.join('\n\n')}\n\nContinue from here.`;
}

// Conversation management functions
function initializeNewSession(profile = null, customPrompt = null) {
    currentSessionId = Date.now().toString();
    currentTranscription = '';
    conversationHistory = [];
    screenAnalysisHistory = [];
    conversationCtxHistory = [];
    currentProfile = profile;
    currentCustomPrompt = customPrompt;
    console.log('New conversation session started:', currentSessionId, 'profile:', profile);

    // Save initial session with profile context
    if (profile) {
        sendToRenderer('save-session-context', {
            sessionId: currentSessionId,
            profile: profile,
            customPrompt: customPrompt || '',
        });
    }
}

function saveConversationTurn(transcription, aiResponse) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const conversationTurn = {
        timestamp: Date.now(),
        transcription: transcription.trim(),
        ai_response: aiResponse.trim(),
    };

    conversationHistory.push(conversationTurn);
    console.log('Saved conversation turn:', conversationTurn);

    // Send to renderer to save in IndexedDB
    sendToRenderer('save-conversation-turn', {
        sessionId: currentSessionId,
        turn: conversationTurn,
        fullHistory: conversationHistory,
    });
}

function saveScreenAnalysis(prompt, response, model) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const analysisEntry = {
        timestamp: Date.now(),
        prompt: prompt,
        response: response.trim(),
        model: model,
    };

    screenAnalysisHistory.push(analysisEntry);
    console.log('Saved screen analysis:', analysisEntry);

    // Send to renderer to save
    sendToRenderer('save-screen-analysis', {
        sessionId: currentSessionId,
        analysis: analysisEntry,
        fullHistory: screenAnalysisHistory,
        profile: currentProfile,
        customPrompt: currentCustomPrompt,
    });
}

function getCurrentSessionData() {
    return {
        sessionId: currentSessionId,
        history: conversationHistory,
    };
}

async function getEnabledTools() {
    const tools = [];

    // Check if Google Search is enabled (default: true)
    const googleSearchEnabled = await getStoredSetting('googleSearchEnabled', 'true');
    console.log('Google Search enabled:', googleSearchEnabled);

    if (googleSearchEnabled === 'true') {
        tools.push({ googleSearch: {} });
        console.log('Added Google Search tool');
    } else {
        console.log('Google Search tool disabled');
    }

    return tools;
}

async function getStoredSetting(key, defaultValue) {
    try {
        const windows = BrowserWindow.getAllWindows();
        if (windows.length > 0) {
            // Wait a bit for the renderer to be ready
            await new Promise(resolve => setTimeout(resolve, 100));

            // Try to get setting from renderer process localStorage
            const value = await windows[0].webContents.executeJavaScript(`
                (function() {
                    try {
                        if (typeof localStorage === 'undefined') {
                            console.log('localStorage not available yet for ${key}');
                            return '${defaultValue}';
                        }
                        const stored = localStorage.getItem('${key}');
                        console.log('Retrieved setting ${key}:', stored);
                        return stored || '${defaultValue}';
                    } catch (e) {
                        console.error('Error accessing localStorage for ${key}:', e);
                        return '${defaultValue}';
                    }
                })()
            `);
            return value;
        }
    } catch (error) {
        console.error('Error getting stored setting for', key, ':', error.message);
    }
    console.log('Using default value for', key, ':', defaultValue);
    return defaultValue;
}

/**
 * Check if a transcription is mostly English / Latin-script text.
 * Returns false for noise markers, non-Latin scripts, and very short gibberish.
 */
function isValidEnglishTranscription(text) {
    const cleaned = text.trim();

    // Skip noise markers from Gemini
    if (/^\[?noise\]?$/i.test(cleaned) || /^<?noise>?$/i.test(cleaned)) {
        return false;
    }

    // Must have at least 3 actual words to be worth processing
    const words = cleaned.split(/\s+/).filter(w => w.length > 0);
    if (words.length < 3) {
        return false;
    }

    // Count Latin-script characters vs non-Latin
    const latinChars = (cleaned.match(/[a-zA-Z0-9\s.,!?'";\-:()]/g) || []).length;
    const totalNonSpace = cleaned.replace(/\s/g, '').length;

    // If less than 50% Latin characters, it's likely non-English noise
    if (totalNonSpace > 0 && latinChars / totalNonSpace < 0.5) {
        return false;
    }

    return true;
}

// Dispatch lock: prevents overlapping API calls and adds cooldown between responses
let isDispatching = false;
let lastDispatchTime = 0;
const DISPATCH_COOLDOWN_MS = 3000; // 3 seconds cooldown after a response completes

/**
 * Check if a transcription is just interviewer filler, not a real question.
 * Filler like "okay let's move on", "go ahead", "next question" shouldn't trigger a response.
 */
function isInterviewerFiller(text) {
    const cleaned = text
        .trim()
        .toLowerCase()
        .replace(/<noise>/g, '')
        .replace(/\[noise\]/g, '')
        .trim();
    const fillerPatterns = [
        /^(okay|ok|so|alright|right|sure|good|great|hmm|um|uh)\s*(,?\s*(let'?s|we)?\s*(go|move|have|proceed|continue|next|start)\s*(on|ahead|forward|with)?)?[\s.!?]*$/i,
        /^(next|another)\s*(question|one)[\s.!?]*$/i,
        /^(go\s*ahead|carry\s*on|proceed)[\s.!?]*$/i,
        /^(i'?m\s*ready|ready|let'?s\s*(go|start|begin))[\s.!?]*$/i,
        /^(okay|ok)\s*(good|great|fine|nice|perfect|wonderful)[\s.!?]*$/i,
        /^(tell\s*me\s*more|go\s*on|continue)[\s.!?]*$/i,
        /^(let'?s\s*have\s*another\s*question)[\s.!?]*$/i,
    ];
    return fillerPatterns.some(p => p.test(cleaned));
}

/**
 * Strip noise markers from transcription before sending to AI.
 */
function cleanTranscriptionForAI(text) {
    return text
        .replace(/<noise>/gi, '')
        .replace(/\[noise\]/gi, '')
        .replace(/\s{2,}/g, ' ')
        .trim();
}

async function dispatchToTextProvider(transcription) {
    if (!transcription || transcription.trim() === '') return;

    // LOCK: If already generating a response, skip this transcription
    if (isDispatching) {
        console.log(`[BLOCKED] Already generating a response, ignoring: "${transcription.trim().substring(0, 60)}..."`);
        return;
    }

    // COOLDOWN: Don't dispatch too quickly after last response
    const timeSinceLastDispatch = Date.now() - lastDispatchTime;
    if (timeSinceLastDispatch < DISPATCH_COOLDOWN_MS) {
        console.log(
            `[COOLDOWN] ${Math.ceil((DISPATCH_COOLDOWN_MS - timeSinceLastDispatch) / 1000)}s remaining, skipping: "${transcription.trim().substring(0, 60)}..."`
        );
        return;
    }

    // Filter out noise and non-English transcriptions
    if (!isValidEnglishTranscription(transcription)) {
        console.log(`[SKIPPED] Non-English or noise detected: "${transcription.trim().substring(0, 80)}"`);
        sendToRenderer('update-status', 'Listening...');
        return;
    }

    // Filter out interviewer filler (not a real question)
    if (isInterviewerFiller(transcription)) {
        console.log(`[FILLER] Interviewer filler, not a question: "${transcription.trim()}"`);
        sendToRenderer('update-status', 'Listening...');
        return;
    }

    // Clean noise markers from the transcription before sending to AI
    const cleanedTranscription = cleanTranscriptionForAI(transcription);
    if (!cleanedTranscription || cleanedTranscription.length < 5) {
        console.log(`[SKIPPED] Transcription too short after cleaning: "${cleanedTranscription}"`);
        sendToRenderer('update-status', 'Listening...');
        return;
    }

    // Set lock
    isDispatching = true;
    const dispatchStartTime = Date.now();

    // Log exactly what the interviewer said for analysis
    console.log('\n' + '='.repeat(80));
    console.log(`[INTERVIEWER] "${cleanedTranscription}"`);
    console.log('='.repeat(80) + '\n');

    const prefs = getPreferences();

    // Determine primary provider and model
    let provider = prefs.textProvider || null;
    let model = prefs.textModel || null;

    // Gather ALL API keys for the failover chain
    const apiKeys = {
        anthropic: getAnthropicApiKey(),
        openrouter: getOpenrouterApiKey(),
        gemini: getApiKey(),
    };

    // Pick primary provider: user's choice, or first available from chain
    if (!provider || !apiKeys[provider]) {
        const chain = ['anthropic', 'openrouter', 'gemini'];
        provider = chain.find(p => apiKeys[p]) || 'gemini';
    }

    // Resolve model for selected provider
    if (!model && PROVIDERS[provider]) {
        model = PROVIDERS[provider].defaultModel;
    }

    try {
        sendToRenderer('update-status', `Thinking (${PROVIDERS[provider]?.name || provider})...`);

        let resolvedProvider = provider;
        let resolvedModel = model;

        const result = await sendToProvider(cleanedTranscription, {
            provider,
            model,
            apiKey: apiKeys[provider] || '',
            apiKeys,
            systemPrompt: currentSystemPrompt || 'You are a helpful assistant.',
            conversationHistory: conversationCtxHistory,
            onToken: (displayText, isFirst) => {
                sendToRenderer(isFirst ? 'new-response' : 'update-response', displayText);
            },
            onDone: cleanedResponse => {
                if (cleanedResponse) {
                    conversationCtxHistory.push({ role: 'user', content: cleanedTranscription });
                    conversationCtxHistory.push({ role: 'assistant', content: cleanedResponse });

                    if (conversationCtxHistory.length > 40) {
                        conversationCtxHistory = conversationCtxHistory.slice(-40);
                    }

                    saveConversationTurn(cleanedTranscription, cleanedResponse);
                }
                sendToRenderer('update-status', 'Listening...');
            },
            onError: errorMsg => {
                console.error('All providers failed:', errorMsg);
                sendToRenderer('update-status', 'All providers failed. Check your API keys.');
            },
            onStatus: statusMsg => {
                sendToRenderer('update-status', statusMsg);
            },
        });

        if (result) {
            resolvedProvider = result.provider || provider;
            resolvedModel = result.model || model;
            const elapsed = ((Date.now() - dispatchStartTime) / 1000).toFixed(1);
            console.log(`[TIMING] Response completed in ${elapsed}s (${resolvedProvider}/${resolvedModel})`);
        }

        return result;
    } finally {
        // ALWAYS release lock and set cooldown, even on error
        isDispatching = false;
        lastDispatchTime = Date.now();
    }
}

function stripThinkingTags(text) {
    return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

async function initializeGeminiSession(apiKey, customPrompt = '', profile = 'interview', language = 'en-US', isReconnect = false) {
    if (isInitializingSession) {
        console.log('Session initialization already in progress');
        return false;
    }

    isInitializingSession = true;
    if (!isReconnect) {
        sendToRenderer('session-initializing', true);
    }

    // Store params for reconnection
    if (!isReconnect) {
        sessionParams = { apiKey, customPrompt, profile, language };
        reconnectAttempts = 0;
    }

    const client = new GoogleGenAI({
        vertexai: false,
        apiKey: apiKey,
        httpOptions: { apiVersion: 'v1alpha' },
    });

    // Get enabled tools first to determine Google Search status
    const enabledTools = await getEnabledTools();
    const googleSearchEnabled = enabledTools.some(tool => tool.googleSearch);

    const prefs = getPreferences();
    const cvContext = prefs.cvContext || '';
    const systemPrompt = getSystemPrompt(profile, customPrompt, googleSearchEnabled, cvContext);
    currentSystemPrompt = systemPrompt; // Store for text providers (Anthropic, OpenRouter, etc.)

    // Gemini Live is used ONLY as a transcription relay.
    // The actual interview responses come from Anthropic/text providers via dispatchToTextProvider().
    const transcriptionPrompt =
        'You are a silent transcription relay. Your ONLY job is to listen to the audio and transcribe what people say. ' +
        'CRITICAL RULES: ' +
        '1. Always transcribe in English ONLY, regardless of what language you think you hear. ' +
        '2. If you cannot understand what was said, output nothing. Do NOT guess or transcribe in other languages. ' +
        '3. Ignore background noise, music, system sounds, and unclear mumbling entirely. ' +
        '4. Do NOT generate any substantive response. When someone finishes speaking, just reply with a single period "." and nothing else. ' +
        '5. Never answer questions, never give advice, never acknowledge what was said beyond the single period.';

    // Initialize new conversation session only on first connect
    if (!isReconnect) {
        initializeNewSession(profile, customPrompt);
    }

    // Model fallback: try newer model first, fall back to older if quota exceeded
    const LIVE_MODELS = ['gemini-3.1-flash-live-preview', 'gemini-2.5-flash-native-audio-preview-09-2025'];
    const modelToUse = liveModelQuotaExhausted ? LIVE_MODELS[1] : LIVE_MODELS[0];
    console.log(`[Live API] Connecting with model: ${modelToUse}`);

    try {
        const session = await client.live.connect({
            model: modelToUse,
            callbacks: {
                onopen: function () {
                    sendToRenderer('update-status', 'Live session connected');
                },
                onmessage: function (message) {
                    // Only log setup/control messages, NOT individual transcription chunks
                    if (
                        message.setupComplete ||
                        message.serverContent?.generationComplete ||
                        message.serverContent?.turnComplete ||
                        message.serverContent?.interrupted ||
                        message.serverContent?.groundingMetadata
                    ) {
                        console.log('----------------', message);
                    }

                    // Handle input transcription (what was spoken)
                    if (message.serverContent?.inputTranscription?.results) {
                        currentTranscription += formatSpeakerResults(message.serverContent.inputTranscription.results);
                        resetTranscriptionDebounce();
                    } else if (message.serverContent?.inputTranscription?.text) {
                        const text = message.serverContent.inputTranscription.text;
                        if (text.trim() !== '') {
                            currentTranscription += text;
                            // Only reset debounce for Latin-script chunks (likely real English speech)
                            // Non-Latin noise (Hindi, Sinhala, etc.) should NOT delay the dispatch
                            if (chunkHasLatinChars(text)) {
                                resetTranscriptionDebounce();
                            }
                        }
                    }

                    // Ignore outputTranscription entirely - Gemini's own output is irrelevant

                    // generationComplete is a backup trigger - debounce usually fires first
                    if (message.serverContent?.generationComplete) {
                        clearTranscriptionDebounce();
                        isCapturingSpeech = false;
                        if (currentTranscription.trim() !== '') {
                            sendToRenderer('update-status', 'Analyzing...');
                            dispatchToTextProvider(currentTranscription);
                            currentTranscription = '';
                        }
                        messageBuffer = '';
                    }

                    if (message.serverContent?.turnComplete) {
                        sendToRenderer('update-status', 'Listening...');
                    }
                },
                onerror: function (e) {
                    console.log('Session error:', e.message);
                    sendToRenderer('update-status', 'Error: ' + e.message);
                },
                onclose: function (e) {
                    console.log('Session closed:', e.reason);

                    // Reset all state in case session died mid-capture
                    isDispatching = false;
                    isCapturingSpeech = false;
                    clearTranscriptionDebounce();

                    // Don't reconnect if user intentionally closed
                    if (isUserClosing) {
                        isUserClosing = false;
                        sendToRenderer('update-status', 'Session closed');
                        return;
                    }

                    // Detect quota/billing errors - don't keep retrying the same model
                    const reason = (e.reason || '').toLowerCase();
                    if (reason.includes('quota') || reason.includes('billing')) {
                        if (!liveModelQuotaExhausted) {
                            // First time: switch to fallback model
                            liveModelQuotaExhausted = true;
                            console.log('[Live API] Quota exceeded on primary model, switching to fallback...');
                            sendToRenderer('update-status', 'Quota hit, switching model...');
                            reconnectAttempts = 0; // Reset attempts for fallback
                            attemptReconnect();
                            return;
                        } else {
                            // Both models quota-exhausted
                            console.log('[Live API] Both models quota-exhausted. Wait for quota reset.');
                            sendToRenderer('update-status', 'API quota exceeded - try again later');
                            return;
                        }
                    }

                    // Attempt reconnection for non-quota errors
                    if (sessionParams && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                        attemptReconnect();
                    } else {
                        sendToRenderer('update-status', 'Session closed');
                    }
                },
            },
            config: {
                responseModalities: [Modality.AUDIO],
                // NOTE: proactiveAudio intentionally omitted - Gemini should NOT jump in unprompted
                // NOTE: outputAudioTranscription intentionally omitted - we don't need Gemini's response text
                // Enable input transcription (what the interviewer/user says)
                inputAudioTranscription: {
                    enableSpeakerDiarization: true,
                    minSpeakerCount: 2,
                    maxSpeakerCount: 2,
                },
                tools: enabledTools,
                contextWindowCompression: { slidingWindow: {} },
                speechConfig: { languageCode: language },
                systemInstruction: {
                    parts: [{ text: transcriptionPrompt }],
                },
            },
        });

        isInitializingSession = false;
        if (!isReconnect) {
            sendToRenderer('session-initializing', false);
        }
        return session;
    } catch (error) {
        console.error('Failed to initialize Gemini session:', error);
        isInitializingSession = false;
        if (!isReconnect) {
            sendToRenderer('session-initializing', false);
        }
        return null;
    }
}

async function attemptReconnect() {
    reconnectAttempts++;
    console.log(`Reconnection attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}`);

    // Clear stale buffers
    messageBuffer = '';
    currentTranscription = '';
    // Don't reset conversationCtxHistory to preserve context across reconnects

    sendToRenderer('update-status', `Reconnecting... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);

    // Wait before attempting
    await new Promise(resolve => setTimeout(resolve, RECONNECT_DELAY));

    try {
        const session = await initializeGeminiSession(
            sessionParams.apiKey,
            sessionParams.customPrompt,
            sessionParams.profile,
            sessionParams.language,
            true // isReconnect
        );

        if (session && global.geminiSessionRef) {
            global.geminiSessionRef.current = session;

            // Restore context from conversation history via text message
            const contextMessage = buildContextMessage();
            if (contextMessage) {
                try {
                    console.log('Restoring conversation context...');
                    await session.sendRealtimeInput({ text: contextMessage });
                } catch (contextError) {
                    console.error('Failed to restore context:', contextError);
                    // Continue without context - better than failing
                }
            }

            // Don't reset reconnectAttempts here - let it reset on next fresh session
            sendToRenderer('update-status', 'Reconnected! Listening...');
            console.log('Session reconnected successfully');
            return true;
        }
    } catch (error) {
        console.error(`Reconnection attempt ${reconnectAttempts} failed:`, error);
    }

    // If we still have attempts left, try again
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        return attemptReconnect();
    }

    // Max attempts reached - notify frontend
    console.log('Max reconnection attempts reached');
    sendToRenderer('reconnect-failed', {
        message: 'Tried 3 times to reconnect. Must be upstream/network issues. Try restarting or download updated app from site.',
    });
    sessionParams = null;
    return false;
}

function killExistingSystemAudioDump() {
    return new Promise(resolve => {
        console.log('Checking for existing SystemAudioDump processes...');

        // Kill any existing SystemAudioDump processes
        const killProc = spawn('pkill', ['-f', 'SystemAudioDump'], {
            stdio: 'ignore',
        });

        killProc.on('close', code => {
            if (code === 0) {
                console.log('Killed existing SystemAudioDump processes');
            } else {
                console.log('No existing SystemAudioDump processes found');
            }
            resolve();
        });

        killProc.on('error', err => {
            console.log('Error checking for existing processes (this is normal):', err.message);
            resolve();
        });

        // Timeout after 2 seconds
        setTimeout(() => {
            killProc.kill();
            resolve();
        }, 2000);
    });
}

async function startMacOSAudioCapture(geminiSessionRef) {
    if (process.platform !== 'darwin') return false;

    // Kill any existing SystemAudioDump processes first
    await killExistingSystemAudioDump();

    console.log('Starting macOS audio capture with SystemAudioDump...');

    const { app } = require('electron');
    const path = require('path');

    let systemAudioPath;
    if (app.isPackaged) {
        systemAudioPath = path.join(process.resourcesPath, 'SystemAudioDump');
    } else {
        systemAudioPath = path.join(__dirname, '../assets', 'SystemAudioDump');
    }

    console.log('SystemAudioDump path:', systemAudioPath);

    const spawnOptions = {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
            ...process.env,
        },
    };

    systemAudioProc = spawn(systemAudioPath, [], spawnOptions);

    if (!systemAudioProc.pid) {
        console.error('Failed to start SystemAudioDump');
        return false;
    }

    console.log('SystemAudioDump started with PID:', systemAudioProc.pid);

    const CHUNK_DURATION = 0.1;
    const SAMPLE_RATE = 24000;
    const BYTES_PER_SAMPLE = 2;
    const CHANNELS = 2;
    const CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_DURATION;

    let audioBuffer = Buffer.alloc(0);

    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);

        while (audioBuffer.length >= CHUNK_SIZE) {
            const chunk = audioBuffer.slice(0, CHUNK_SIZE);
            audioBuffer = audioBuffer.slice(CHUNK_SIZE);

            const monoChunk = CHANNELS === 2 ? convertStereoToMono(chunk) : chunk;

            if (currentProviderMode === 'cloud') {
                sendCloudAudio(monoChunk);
            } else {
                const base64Data = monoChunk.toString('base64');
                sendAudioToGemini(base64Data, geminiSessionRef);
            }

            if (process.env.DEBUG_AUDIO) {
                console.log(`Processed audio chunk: ${chunk.length} bytes`);
                saveDebugAudio(monoChunk, 'system_audio');
            }
        }

        const maxBufferSize = SAMPLE_RATE * BYTES_PER_SAMPLE * 1;
        if (audioBuffer.length > maxBufferSize) {
            audioBuffer = audioBuffer.slice(-maxBufferSize);
        }
    });

    systemAudioProc.stderr.on('data', data => {
        console.error('SystemAudioDump stderr:', data.toString());
    });

    systemAudioProc.on('close', code => {
        console.log('SystemAudioDump process closed with code:', code);
        systemAudioProc = null;
    });

    systemAudioProc.on('error', err => {
        console.error('SystemAudioDump process error:', err);
        systemAudioProc = null;
    });

    return true;
}

function convertStereoToMono(stereoBuffer) {
    const samples = stereoBuffer.length / 4;
    const monoBuffer = Buffer.alloc(samples * 2);

    for (let i = 0; i < samples; i++) {
        const leftSample = stereoBuffer.readInt16LE(i * 4);
        monoBuffer.writeInt16LE(leftSample, i * 2);
    }

    return monoBuffer;
}

function stopMacOSAudioCapture() {
    if (systemAudioProc) {
        console.log('Stopping SystemAudioDump...');
        systemAudioProc.kill('SIGTERM');
        systemAudioProc = null;
    }
}

async function sendAudioToGemini(base64Data, geminiSessionRef) {
    if (!geminiSessionRef.current) return;

    try {
        process.stdout.write('.');
        await geminiSessionRef.current.sendRealtimeInput({
            audio: {
                data: base64Data,
                mimeType: 'audio/pcm;rate=24000',
            },
        });
    } catch (error) {
        console.error('Error sending audio to Gemini:', error);
    }
}

async function sendImageWithFailover(base64Data, prompt, isMultiCapture = false) {
    const prefs = getPreferences();
    const provider = prefs.textProvider || 'gemini';
    const model = prefs.textModel || 'gemini-2.5-flash';

    const apiKeys = {
        anthropic: getAnthropicApiKey() || '',
        openrouter: getOpenrouterApiKey() || '',
        gemini: getApiKey() || '',
    };

    // For screen analysis, prefer Anthropic Sonnet (highest vision quality)
    // Override the default model to Sonnet when Anthropic key is available
    let imageProvider = provider;
    let imageModel = model;
    if (apiKeys.anthropic && provider !== 'anthropic') {
        imageProvider = 'anthropic';
        imageModel = 'claude-sonnet-4-6';
    }

    const chain = buildFailoverChain(imageProvider, imageModel, apiKeys);

    for (let i = 0; i < chain.length; i++) {
        const step = chain[i];
        const isLast = i === chain.length - 1;

        // Find the best vision-capable model for this provider
        const providerConfig = PROVIDERS[step.provider];
        const selectedModel = providerConfig?.models?.find(m => m.id === step.model);
        let visionModel = step.model;

        if (selectedModel && !selectedModel.vision) {
            // User's model doesn't support vision, find one that does
            const visionFallback = providerConfig.models.find(m => m.vision);
            if (!visionFallback) {
                console.log(`[Image] Skipping ${step.provider} (no vision models)`);
                continue;
            }
            visionModel = visionFallback.id;
            console.log(`[Image] ${step.model} has no vision, swapping to ${visionModel}`);
        }

        try {
            console.log(`[Image] Trying ${step.provider}/${visionModel}...`);
            sendToRenderer('update-status', `Analyzing with ${step.provider}/${visionModel}...`);

            let result;
            if (step.provider === 'anthropic') {
                result = await sendImageViaAnthropic(base64Data, prompt, visionModel, step.apiKey);
            } else if (step.provider === 'gemini') {
                result = await sendImageViaGeminiSDK(base64Data, prompt, visionModel, step.apiKey);
            } else if (step.provider === 'openrouter') {
                result = await sendImageViaOpenAI(base64Data, prompt, visionModel, step.apiKey, step.provider);
            }

            if (result && result.success) {
                saveScreenAnalysis(prompt, result.text, `${step.provider}/${visionModel}`);
                return result;
            }
            throw new Error(result?.error || 'Unknown image analysis error');
        } catch (error) {
            console.error(`[Image] ${step.provider}/${visionModel} failed: ${error.message}`);

            // Gemini model-level failover: try other models in the same provider
            // Each Gemini model has its own daily quota, so flash-lite may still work
            if (step.provider === 'gemini' && error.message.includes('429')) {
                const otherModels = providerConfig.models.filter(m => m.vision && m.id !== visionModel);
                for (const alt of otherModels) {
                    try {
                        console.log(`[Image] Gemini quota hit, trying alternate: ${alt.id}...`);
                        sendToRenderer('update-status', `Quota hit, trying ${alt.name}...`);
                        const altResult = await sendImageViaGeminiSDK(base64Data, prompt, alt.id, step.apiKey);
                        if (altResult && altResult.success) {
                            saveScreenAnalysis(prompt, altResult.text, `gemini/${alt.id}`);
                            return altResult;
                        }
                    } catch (altError) {
                        console.error(`[Image] gemini/${alt.id} also failed: ${altError.message}`);
                    }
                }
            }

            // OpenRouter model-level failover: when paid model 402s, try free models
            if (step.provider === 'openrouter' && (error.message.includes('402') || error.message.includes('429'))) {
                const freeModels = providerConfig.models.filter(m => m.vision && m.id !== visionModel && m.id.includes(':free'));
                for (const alt of freeModels) {
                    try {
                        console.log(`[Image] OpenRouter credits exhausted, trying free: ${alt.id}...`);
                        sendToRenderer('update-status', `Credits low, trying ${alt.name}...`);
                        const altResult = await sendImageViaOpenAI(base64Data, prompt, alt.id, step.apiKey, 'openrouter');
                        if (altResult && altResult.success) {
                            saveScreenAnalysis(prompt, altResult.text, `openrouter/${alt.id}`);
                            return altResult;
                        }
                    } catch (altError) {
                        console.error(`[Image] openrouter/${alt.id} also failed: ${altError.message}`);
                    }
                }
            }

            if (isLast) {
                sendToRenderer('update-status', 'All image providers failed');
                return { success: false, error: `All providers failed. Last: ${error.message}` };
            }
        }
    }
    return { success: false, error: 'No providers available' };
}
// Anthropic Claude vision (supports single or multiple images)
async function sendImageViaAnthropic(base64Data, prompt, model, apiKey) {
    const images = Array.isArray(base64Data) ? base64Data : [base64Data];
    const timeoutMs = (images.length > 1 ? CLOUD_TIMEOUT_MS * 2 : CLOUD_TIMEOUT_MS) || 30000;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const systemPrompt = currentSystemPrompt || 'You are a helpful assistant.';

        const contentParts = [];
        for (const img of images) {
            contentParts.push({
                type: 'image',
                source: {
                    type: 'base64',
                    media_type: 'image/jpeg',
                    data: img,
                },
            });
        }
        contentParts.push({ type: 'text', text: prompt });

        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
            },
            body: JSON.stringify({
                model,
                max_tokens: 4096,
                system: systemPrompt,
                messages: [{ role: 'user', content: contentParts }],
                stream: true,
            }),
            signal: controller.signal,
        });

        clearTimeout(timer);

        if (!response.ok) {
            const err = await response.text();
            throw new Error(`Anthropic API ${response.status}: ${err.substring(0, 200)}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let isFirst = true;
        let leftover = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = leftover + decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            leftover = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const json = JSON.parse(line.slice(6));
                        if (json.type === 'content_block_delta' && json.delta?.text) {
                            fullText += json.delta.text;
                            sendToRenderer(isFirst ? 'new-response' : 'update-response', fullText);
                            isFirst = false;
                        }
                    } catch (_e) {
                        // skip
                    }
                }
            }
        }

        if (fullText.trim()) {
            return { success: true, text: fullText.trim(), model };
        }
        return { success: false, error: 'Empty response from Anthropic' };
    } catch (error) {
        clearTimeout(timer);
        throw error;
    }
}

// Gemini SDK vision (supports single or multiple images)
async function sendImageViaGeminiSDK(base64Data, prompt, model, apiKey) {
    const images = Array.isArray(base64Data) ? base64Data : [base64Data];
    const ai = new GoogleGenAI({ apiKey });
    const timeoutMs = (images.length > 1 ? CLOUD_TIMEOUT_MS * 2 : CLOUD_TIMEOUT_MS) || 30000;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const contents = [];
        for (const img of images) {
            contents.push({ inlineData: { mimeType: 'image/jpeg', data: img } });
        }
        contents.push({ text: prompt });

        console.log(`Sending image to ${model} (streaming)...`);
        const response = await ai.models.generateContentStream({ model, contents });

        let fullText = '';
        let isFirst = true;
        for await (const chunk of response) {
            if (chunk.text) {
                fullText += chunk.text;
                sendToRenderer(isFirst ? 'new-response' : 'update-response', fullText);
                isFirst = false;
            }
        }

        clearTimeout(timer);
        console.log(`Image response completed from ${model}`);
        return { success: true, text: fullText, model };
    } catch (error) {
        clearTimeout(timer);
        throw error;
    }
}

// OpenRouter vision (OpenAI-compatible) - supports single or multiple images
async function sendImageViaOpenAI(base64Data, prompt, model, apiKey, provider) {
    const images = Array.isArray(base64Data) ? base64Data : [base64Data];
    const baseUrl = 'https://openrouter.ai/api/v1/chat/completions';
    const timeoutMs = (images.length > 1 ? CLOUD_TIMEOUT_MS * 2 : CLOUD_TIMEOUT_MS) || 30000;

    const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
    };
    if (provider === 'openrouter') {
        headers['HTTP-Referer'] = 'https://github.com/AswaniSahoo/KAITE';
        headers['X-Title'] = 'KAITE';
    }

    const contentParts = [];
    for (const img of images) {
        contentParts.push({ type: 'image_url', image_url: { url: `data:image/jpeg;base64,${img}` } });
    }
    contentParts.push({ type: 'text', text: prompt });

    const body = {
        model,
        messages: [{ role: 'user', content: contentParts }],
        stream: true,
        max_tokens: 2048,
    };

    const response = await fetch(baseUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
        const errText = await response.text().catch(() => '');
        throw new Error(`HTTP ${response.status}: ${errText.substring(0, 200)}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let isFirst = true;
    let leftover = ''; // Buffer for incomplete SSE lines across chunk boundaries
    let streamDone = false;

    while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = leftover + decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        // Last element may be incomplete - save it for the next chunk
        leftover = lines.pop() || '';

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;

            const data = trimmed.slice(6);
            if (data === '[DONE]') {
                streamDone = true;
                break;
            }
            try {
                const parsed = JSON.parse(data);
                const delta = parsed.choices?.[0]?.delta?.content;
                if (delta) {
                    fullText += delta;
                    const displayText = stripThinkingTags(fullText);
                    sendToRenderer(isFirst ? 'new-response' : 'update-response', displayText);
                    isFirst = false;
                }
            } catch {
                // skip malformed JSON chunks
            }
        }
    }

    if (!fullText) throw new Error('Empty response from vision API');
    console.log(`Image response completed from ${provider}/${model}`);
    return { success: true, text: stripThinkingTags(fullText), model };
}

function setupGeminiIpcHandlers(geminiSessionRef) {
    // Store the geminiSessionRef globally for reconnection access
    global.geminiSessionRef = geminiSessionRef;

    ipcMain.handle('initialize-cloud', async (event, token, profile, userContext) => {
        try {
            currentProviderMode = 'cloud';
            initializeNewSession(profile);
            setOnTurnComplete((transcription, response) => {
                saveConversationTurn(transcription, response);
            });
            sendToRenderer('session-initializing', true);
            await connectCloud(token, profile, userContext);
            sendToRenderer('session-initializing', false);
            return true;
        } catch (err) {
            console.error('[Cloud] Init error:', err);
            currentProviderMode = 'byok';
            sendToRenderer('session-initializing', false);
            return false;
        }
    });

    ipcMain.handle('initialize-gemini', async (event, apiKey, customPrompt, profile = 'interview', language = 'en-US') => {
        currentProviderMode = 'byok';
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language);
        if (session) {
            geminiSessionRef.current = session;
            return true;
        }
        return false;
    });

    ipcMain.handle('send-audio-content', async (event, { data, mimeType }) => {
        if (currentProviderMode === 'cloud') {
            try {
                const pcmBuffer = Buffer.from(data, 'base64');
                sendCloudAudio(pcmBuffer);
                return { success: true };
            } catch (error) {
                console.error('Error sending cloud audio:', error);
                return { success: false, error: error.message };
            }
        }
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };
        try {
            process.stdout.write('.');
            await geminiSessionRef.current.sendRealtimeInput({
                audio: { data: data, mimeType: mimeType },
            });
            return { success: true };
        } catch (error) {
            console.error('Error sending system audio:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-mic-audio-content', async (event, { data, mimeType }) => {
        if (currentProviderMode === 'cloud') {
            try {
                const pcmBuffer = Buffer.from(data, 'base64');
                sendCloudAudio(pcmBuffer);
                return { success: true };
            } catch (error) {
                console.error('Error sending cloud mic audio:', error);
                return { success: false, error: error.message };
            }
        }
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };
        try {
            process.stdout.write(',');
            await geminiSessionRef.current.sendRealtimeInput({
                audio: { data: data, mimeType: mimeType },
            });
            return { success: true };
        } catch (error) {
            console.error('Error sending mic audio:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-image-content', async (event, { data, prompt, isMultiCapture }) => {
        try {
            // Multi-capture: data is an array of base64 strings
            if (isMultiCapture && Array.isArray(data)) {
                console.log(`Multi-capture: processing ${data.length} images`);

                // Validate all images
                for (let i = 0; i < data.length; i++) {
                    if (!data[i] || typeof data[i] !== 'string') {
                        return { success: false, error: `Invalid image data at index ${i}` };
                    }
                    const buf = Buffer.from(data[i], 'base64');
                    if (buf.length < 1000) {
                        return { success: false, error: `Image ${i + 1} too small (${buf.length} bytes)` };
                    }
                }

                if (currentProviderMode === 'cloud') {
                    const sent = sendCloudImage(data[data.length - 1]);
                    if (!sent) return { success: false, error: 'Cloud connection not active' };
                    return { success: true, model: 'cloud' };
                }

                const result = await sendImageWithFailover(data, prompt, true);
                return result;
            }

            // Single image: original behavior
            if (!data || typeof data !== 'string') {
                console.error('Invalid image data received');
                return { success: false, error: 'Invalid image data' };
            }

            const buffer = Buffer.from(data, 'base64');

            if (buffer.length < 1000) {
                console.error(`Image buffer too small: ${buffer.length} bytes`);
                return { success: false, error: 'Image buffer too small' };
            }

            process.stdout.write('!');

            if (currentProviderMode === 'cloud') {
                const sent = sendCloudImage(data);
                if (!sent) {
                    return { success: false, error: 'Cloud connection not active' };
                }
                return { success: true, model: 'cloud' };
            }

            // Use failover-aware image analysis
            const result = await sendImageWithFailover(data, prompt);
            return result;
        } catch (error) {
            console.error('Error sending image:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-text-message', async (event, text) => {
        if (!text || typeof text !== 'string' || text.trim().length === 0) {
            return { success: false, error: 'Invalid text message' };
        }

        if (currentProviderMode === 'cloud') {
            try {
                console.log('Sending text to cloud:', text);
                sendCloudText(text.trim());
                return { success: true };
            } catch (error) {
                console.error('Error sending cloud text:', error);
                return { success: false, error: error.message };
            }
        }

        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };

        try {
            console.log('Sending text message:', text);

            dispatchToTextProvider(text.trim());

            await geminiSessionRef.current.sendRealtimeInput({ text: text.trim() });
            return { success: true };
        } catch (error) {
            console.error('Error sending text:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-macos-audio', async event => {
        if (process.platform !== 'darwin') {
            return {
                success: false,
                error: 'macOS audio capture only available on macOS',
            };
        }

        try {
            const success = await startMacOSAudioCapture(geminiSessionRef);
            return { success };
        } catch (error) {
            console.error('Error starting macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('stop-macos-audio', async event => {
        try {
            stopMacOSAudioCapture();
            return { success: true };
        } catch (error) {
            console.error('Error stopping macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('close-session', async event => {
        try {
            stopMacOSAudioCapture();

            if (currentProviderMode === 'cloud') {
                closeCloud();
                currentProviderMode = 'byok';
                return { success: true };
            }

            // Set flag to prevent reconnection attempts
            isUserClosing = true;
            sessionParams = null;

            // Cleanup session
            if (geminiSessionRef.current) {
                await geminiSessionRef.current.close();
                geminiSessionRef.current = null;
            }

            return { success: true };
        } catch (error) {
            console.error('Error closing session:', error);
            return { success: false, error: error.message };
        }
    });

    // Conversation history IPC handlers
    ipcMain.handle('get-current-session', async event => {
        try {
            return { success: true, data: getCurrentSessionData() };
        } catch (error) {
            console.error('Error getting current session:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-new-session', async event => {
        try {
            initializeNewSession();
            return { success: true, sessionId: currentSessionId };
        } catch (error) {
            console.error('Error starting new session:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('update-google-search-setting', async (event, enabled) => {
        try {
            console.log('Google Search setting updated to:', enabled);
            // The setting is already saved in localStorage by the renderer
            // This is just for logging/confirmation
            return { success: true };
        } catch (error) {
            console.error('Error updating Google Search setting:', error);
            return { success: false, error: error.message };
        }
    });
}

module.exports = {
    initializeGeminiSession,
    getEnabledTools,
    getStoredSetting,
    sendToRenderer,
    initializeNewSession,
    saveConversationTurn,
    getCurrentSessionData,
    killExistingSystemAudioDump,
    startMacOSAudioCapture,
    convertStereoToMono,
    stopMacOSAudioCapture,
    sendAudioToGemini,
    sendImageWithFailover,
    setupGeminiIpcHandlers,
    formatSpeakerResults,
};
