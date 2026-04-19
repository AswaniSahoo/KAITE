/**
 * Unified Text Response Provider System with Cascading Failover
 *
 * Strategy: Try the primary provider/model. If it fails (timeout, billing,
 * server error, anything), automatically cascade to the next provider in
 * the failover chain. The chain is ordered by quality and reliability:
 *
 *   1. Primary (user-selected or auto-detected)
 *   2. Gemini (Google SDK, free tier, vision + thinking)
 *   3. OpenRouter (free Gemma 4 models, paid premium)
 *   4. Local Ollama (offline fallback, never fails on billing)
 *
 * Every individual provider call gets:
 *   - AbortController timeout (30s cloud, 60s local)
 *   - No retry on 4xx (auth/billing won't self-fix)
 *   - Single retry on 5xx/timeout before cascading to next provider
 */

const { GoogleGenAI } = require('@google/genai');

// ── Provider & Model Registry ──────────────────────────────────────────────
// Models are ordered by performance/quality within each provider.
// The first model in each list is the default (best bang for buck).

const PROVIDERS = {
    gemini: {
        name: 'Gemini (Google)',
        baseUrl: null, // uses SDK, not raw HTTP
        keyField: 'geminiApiKey',
        models: [
            // All free (15 RPM on free tier), all have vision + thinking
            { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash (Free)', contextWindow: 1048576, speed: 'fast', vision: true },
            { id: 'gemini-2.5-pro-preview-05-06', name: 'Gemini 2.5 Pro (Free)', contextWindow: 1048576, speed: 'medium', vision: true },
            { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro (Free)', contextWindow: 1048576, speed: 'medium', vision: true },
        ],
        defaultModel: 'gemini-2.5-flash',
    },
    openrouter: {
        name: 'OpenRouter',
        baseUrl: 'https://openrouter.ai/api/v1/chat/completions',
        keyField: 'openrouterApiKey',
        models: [
            // Free vision models (no credits needed, best for screen analysis)
            { id: 'google/gemma-4-31b-it:free', name: 'Gemma 4 31B (Free, Vision)', contextWindow: 262144, speed: 'fast', vision: true },
            { id: 'google/gemma-4-27b-a4b-it:free', name: 'Gemma 4 26B MoE (Free, Fast)', contextWindow: 262144, speed: 'fastest', vision: true },
            // Premium vision + thinking models (accurate, reliable)
            { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4 (Paid)', contextWindow: 200000, speed: 'medium', vision: true },
            { id: 'google/gemini-2.5-flash-preview', name: 'Gemini 2.5 Flash (Paid)', contextWindow: 1048576, speed: 'fast', vision: true },
            { id: 'openai/gpt-4.1-mini', name: 'GPT-4.1 Mini (Paid)', contextWindow: 1047576, speed: 'fast', vision: true },
            { id: 'deepseek/deepseek-r1', name: 'DeepSeek R1 (Thinking)', contextWindow: 163840, speed: 'medium', vision: false },
        ],
        defaultModel: 'google/gemma-4-31b-it:free',
    },
    ollama: {
        name: 'Ollama (Local)',
        baseUrl: null, // resolved at runtime from preferences
        keyField: null, // no API key needed
        models: [
            { id: 'gemma4:latest', name: 'Gemma 4 (9.6GB, Vision)', contextWindow: 131072, speed: 'medium', vision: true },
            { id: 'gemma3:12b', name: 'Gemma 3 12B (Vision)', contextWindow: 131072, speed: 'medium', vision: true },
        ],
        defaultModel: 'gemma4:latest',
    },
};

// ── Timeout + Retry Configuration ──────────────────────────────────────────

const CLOUD_TIMEOUT_MS = 30000; // 30 seconds for cloud APIs
const LOCAL_TIMEOUT_MS = 60000; // 60 seconds for local Ollama (slower)
const MAX_RETRIES = 2; // 2 attempts per provider, then cascade
const BASE_RETRY_DELAY_MS = 1000;

// ── Utility ────────────────────────────────────────────────────────────────

function stripThinkingTags(text) {
    return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

function trimConversationHistory(history, maxChars = 42000) {
    if (!history || history.length === 0) return [];
    let totalChars = 0;
    const trimmed = [];

    for (let i = history.length - 1; i >= 0; i--) {
        const turn = history[i];
        const turnChars = (turn.content || '').length;
        if (totalChars + turnChars > maxChars) break;
        totalChars += turnChars;
        trimmed.unshift(turn);
    }
    return trimmed;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function isNonRetryableError(errorMsg) {
    // 4xx errors: auth, billing, forbidden, not found, rate limit
    return errorMsg && /API (40[0-9]|4[1-9][0-9])/.test(errorMsg);
}

// ── OpenAI-Compatible Streaming (Groq, OpenRouter) ────────────────────────

async function streamOpenAICompatible({ baseUrl, apiKey, model, messages, onToken, onDone, onError, providerName }) {
    let lastError = null;
    const timeoutMs = CLOUD_TIMEOUT_MS;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const headers = {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            };

            // OpenRouter requires extra headers
            if (providerName === 'OpenRouter') {
                headers['HTTP-Referer'] = 'https://github.com/AswaniSahoo/KAITE';
                headers['X-Title'] = 'KAITE';
            }

            const response = await fetch(baseUrl, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    model,
                    messages,
                    stream: true,
                    temperature: 0.7,
                    max_tokens: 4096,
                }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`${providerName} API ${response.status}: ${errorText.substring(0, 200)}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let isFirst = true;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim() !== '');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;

                        try {
                            const json = JSON.parse(data);
                            const token = json.choices?.[0]?.delta?.content || '';
                            if (token) {
                                fullText += token;
                                const displayText = stripThinkingTags(fullText);
                                if (displayText) {
                                    onToken(displayText, isFirst);
                                    isFirst = false;
                                }
                            }
                        } catch (_parseError) {
                            // Skip invalid JSON chunks
                        }
                    }
                }
            }

            const cleanedResponse = stripThinkingTags(fullText);
            onDone(cleanedResponse);
            return cleanedResponse;
        } catch (error) {
            clearTimeout(timeoutId);
            lastError = error;

            const isTimeout = error.name === 'AbortError';
            const errorMsg = isTimeout ? `${providerName} timed out (${timeoutMs / 1000}s)` : error.message;

            // 4xx errors won't fix themselves, bail immediately
            if (isNonRetryableError(error.message)) {
                console.error(`[${providerName}] Non-retryable: ${errorMsg.substring(0, 100)}`);
                break;
            }

            console.error(`[${providerName}] Attempt ${attempt}/${MAX_RETRIES} failed: ${errorMsg}`);

            if (attempt < MAX_RETRIES) {
                const delay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt - 1);
                console.log(`[${providerName}] Retrying in ${delay}ms...`);
                await sleep(delay);
            }
        }
    }

    // All retries exhausted for this provider
    const finalError = `${providerName} failed: ${lastError?.message || 'Unknown error'}`;
    console.error(finalError);
    onError(finalError);
    return null;
}

// ── Gemini SDK Streaming ──────────────────────────────────────────────────

async function streamGeminiSDK({ apiKey, model, messages, systemPrompt, onToken, onDone, onError }) {
    let lastError = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        let timeoutTimer;
        try {
            const ai = new GoogleGenAI({ apiKey });

            const geminiMessages = messages.map(msg => ({
                role: msg.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: msg.content }],
            }));

            const messagesWithSystem = [
                { role: 'user', parts: [{ text: systemPrompt || 'You are a helpful assistant.' }] },
                { role: 'model', parts: [{ text: 'Understood. I will follow these instructions.' }] },
                ...geminiMessages,
            ];

            const timeoutPromise = new Promise((_, reject) => {
                timeoutTimer = setTimeout(() => reject(new Error(`Gemini timed out (${CLOUD_TIMEOUT_MS / 1000}s)`)), CLOUD_TIMEOUT_MS);
            });

            const streamPromise = (async () => {
                const response = await ai.models.generateContentStream({
                    model,
                    contents: messagesWithSystem,
                });

                let fullText = '';
                let isFirst = true;

                for await (const chunk of response) {
                    const chunkText = chunk.text;
                    if (chunkText) {
                        fullText += chunkText;
                        onToken(fullText, isFirst);
                        isFirst = false;
                    }
                }

                return fullText;
            })();

            const fullText = await Promise.race([streamPromise, timeoutPromise]);
            clearTimeout(timeoutTimer);

            if (fullText && fullText.trim()) {
                onDone(fullText.trim());
                return fullText.trim();
            }

            onDone('');
            return '';
        } catch (error) {
            clearTimeout(timeoutTimer);
            lastError = error;
            console.error(`[Gemini] Attempt ${attempt}/${MAX_RETRIES} failed: ${error.message}`);

            if (attempt < MAX_RETRIES) {
                const delay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt - 1);
                console.log(`[Gemini] Retrying in ${delay}ms...`);
                await sleep(delay);
            }
        }
    }

    const finalError = `Gemini failed: ${lastError?.message || 'Unknown error'}`;
    console.error(finalError);
    onError(finalError);
    return null;
}

// ── Ollama Local Streaming ────────────────────────────────────────────────

async function streamOllama({ host, model, messages, onToken, onDone, onError }) {
    const baseUrl = `${host || 'http://127.0.0.1:11434'}/api/chat`;
    let lastError = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), LOCAL_TIMEOUT_MS);

        try {
            const response = await fetch(baseUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    messages,
                    stream: true,
                }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Ollama ${response.status}: ${errorText.substring(0, 200)}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let isFirst = true;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim() !== '');

                for (const line of lines) {
                    try {
                        const json = JSON.parse(line);
                        const token = json.message?.content || '';
                        if (token) {
                            fullText += token;
                            onToken(fullText, isFirst);
                            isFirst = false;
                        }
                    } catch (_parseError) {
                        // Skip invalid JSON
                    }
                }
            }

            if (fullText.trim()) {
                onDone(fullText.trim());
                return fullText.trim();
            }

            onDone('');
            return '';
        } catch (error) {
            clearTimeout(timeoutId);
            lastError = error;

            const isTimeout = error.name === 'AbortError';
            const errorMsg = isTimeout ? `Ollama timed out (${LOCAL_TIMEOUT_MS / 1000}s)` : error.message;
            console.error(`[Ollama] Attempt ${attempt}/${MAX_RETRIES} failed: ${errorMsg}`);

            if (attempt < MAX_RETRIES) {
                await sleep(BASE_RETRY_DELAY_MS);
            }
        }
    }

    const finalError = `Ollama failed: ${lastError?.message || 'Unknown error'}`;
    console.error(finalError);
    onError(finalError);
    return null;
}

// ── Single Provider Dispatch ──────────────────────────────────────────────

async function callSingleProvider(provider, model, apiKey, messages, systemPrompt, onToken, onDone, onError, ollamaHost) {
    const providerConfig = PROVIDERS[provider];
    if (!providerConfig) {
        onError(`Unknown provider: ${provider}`);
        return null;
    }

    if (provider === 'ollama') {
        return streamOllama({
            host: ollamaHost,
            model,
            messages,
            onToken,
            onDone,
            onError,
        });
    }

    if (provider === 'gemini') {
        const historyMsgs = messages.filter(m => m.role !== 'system');
        return streamGeminiSDK({
            apiKey,
            model,
            messages: historyMsgs,
            systemPrompt,
            onToken,
            onDone,
            onError,
        });
    }

    // OpenAI-compatible (Groq, OpenRouter)
    return streamOpenAICompatible({
        baseUrl: providerConfig.baseUrl,
        apiKey,
        model,
        messages,
        onToken,
        onDone,
        onError,
        providerName: providerConfig.name,
    });
}

// ── Cascading Failover Dispatch ───────────────────────────────────────────

/**
 * Build the failover chain. The primary provider goes first, then all
 * others that have valid API keys, ending with local Ollama.
 *
 * @param {string} primaryProvider - User's selected provider
 * @param {object} apiKeys - { groq, openrouter, gemini } key strings
 * @param {string} ollamaHost - Ollama host URL
 * @returns {Array<{provider, model, apiKey}>} Ordered failover chain
 */
function buildFailoverChain(primaryProvider, primaryModel, apiKeys, ollamaHost) {
    const chain = [];

    // Priority order for failover (Gemini first = free + reliable)
    const providerOrder = [
        { key: 'gemini', model: PROVIDERS.gemini.defaultModel },
        { key: 'openrouter', model: PROVIDERS.openrouter.defaultModel },
        { key: 'ollama', model: PROVIDERS.ollama.defaultModel },
    ];

    // Primary goes first with user-selected model
    if (primaryProvider && primaryProvider !== 'ollama') {
        const key = apiKeys[primaryProvider] || '';
        if (key) {
            chain.push({
                provider: primaryProvider,
                model: primaryModel || PROVIDERS[primaryProvider]?.defaultModel,
                apiKey: key,
            });
        }
    }

    if (primaryProvider === 'ollama') {
        chain.push({
            provider: 'ollama',
            model: primaryModel || PROVIDERS.ollama.defaultModel,
            apiKey: null,
        });
    }

    // Add remaining providers as fallbacks (skip the primary since it's already first)
    for (const entry of providerOrder) {
        if (entry.key === primaryProvider) continue; // already in chain

        if (entry.key === 'ollama') {
            // Ollama always available as last resort (no key needed)
            chain.push({ provider: 'ollama', model: entry.model, apiKey: null });
            continue;
        }

        const key = apiKeys[entry.key] || '';
        if (key) {
            chain.push({ provider: entry.key, model: entry.model, apiKey: key });
        }
    }

    return chain;
}

/**
 * Send transcription with cascading failover.
 * Tries each provider in the chain until one succeeds.
 *
 * @param {string} transcription - Text to send
 * @param {object} opts - All options
 * @returns {Promise<{text: string|null, provider: string, model: string}>}
 */
async function sendToProvider(transcription, opts) {
    const {
        provider: primaryProvider,
        model: primaryModel,
        apiKey: primaryApiKey,
        apiKeys, // { groq, openrouter, gemini }
        systemPrompt,
        conversationHistory,
        onToken,
        onDone,
        onError,
        onStatus, // optional: called with status updates
        ollamaHost,
    } = opts;

    if (!transcription || transcription.trim() === '') {
        console.log('Empty transcription, skipping');
        return { text: null, provider: null, model: null };
    }

    // Build failover chain
    const allKeys = apiKeys || {
        openrouter: '',
        gemini: '',
        [primaryProvider]: primaryApiKey || '',
    };

    const chain = buildFailoverChain(primaryProvider, primaryModel, allKeys, ollamaHost);

    if (chain.length === 0) {
        const msg = 'No API keys configured and Ollama not available';
        onError(msg);
        return { text: null, provider: null, model: null };
    }

    // Build messages array
    const messages = [
        { role: 'system', content: systemPrompt || 'You are a helpful assistant.' },
        ...trimConversationHistory(conversationHistory, 42000),
        { role: 'user', content: transcription.trim() },
    ];

    // Try each provider in the chain
    for (let i = 0; i < chain.length; i++) {
        const { provider, model, apiKey } = chain[i];
        const isLast = i === chain.length - 1;

        const statusFn = onStatus || (() => {});
        statusFn(`Thinking (${PROVIDERS[provider]?.name || provider} / ${model})...`);

        console.log(`[Failover ${i + 1}/${chain.length}] Trying ${provider}/${model}...`);

        let succeeded = false;
        let resultText = null;

        const result = await callSingleProvider(
            provider,
            model,
            apiKey,
            messages,
            systemPrompt,
            // onToken: forward to caller
            (displayText, isFirst) => {
                onToken(displayText, isFirst);
            },
            // onDone: mark success
            cleanedResponse => {
                succeeded = true;
                resultText = cleanedResponse;
            },
            // onError: log but don't propagate unless it's the last provider
            errorMsg => {
                console.error(`[Failover] ${provider}/${model} failed: ${errorMsg}`);
                if (isLast) {
                    onError(`All providers failed. Last error: ${errorMsg}`);
                }
            },
            ollamaHost
        );

        if (succeeded && resultText) {
            console.log(`[Failover] Success with ${provider}/${model}`);
            onDone(resultText);
            return { text: resultText, provider, model };
        }

        // If we got here, this provider failed. Try the next one.
        if (!isLast) {
            console.log(`[Failover] Cascading to next provider...`);
            statusFn(`${PROVIDERS[provider]?.name} failed, trying next...`);
            // Brief pause before switching
            await sleep(300);
        }
    }

    return { text: null, provider: null, model: null };
}

// ── Exports ───────────────────────────────────────────────────────────────

module.exports = {
    PROVIDERS,
    sendToProvider,
    buildFailoverChain,
    trimConversationHistory,
    stripThinkingTags,
    CLOUD_TIMEOUT_MS,
    LOCAL_TIMEOUT_MS,
    MAX_RETRIES,
};
