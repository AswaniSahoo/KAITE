/**
 * Unified Text Response Provider System with Cascading Failover
 *
 * Provider chain (ordered by quality/reliability):
 *   1. Anthropic (Claude - fast, high quality, paid)
 *   2. OpenRouter (free Gemma 4 models + paid premium)
 *   3. Gemini (Google SDK, free tier, vision + thinking)
 *
 * Every individual provider call gets:
 *   - AbortController timeout (15s cloud)
 *   - No retry on 4xx (auth/billing won't self-fix)
 *   - Single retry on 5xx/timeout before cascading to next provider
 */

const { GoogleGenAI } = require('@google/genai');

// ── Provider & Model Registry ──────────────────────────────────────────────

const PROVIDERS = {
    anthropic: {
        name: 'Anthropic (Claude)',
        baseUrl: 'https://api.anthropic.com/v1/messages',
        keyField: 'anthropicApiKey',
        models: [
            { id: 'claude-haiku-4-5-20251001', name: 'Haiku 4.5 (Fast, Cheap)', contextWindow: 200000, speed: 'fastest', vision: true },
            { id: 'claude-sonnet-4-6', name: 'Sonnet 4.6 (Quality)', contextWindow: 200000, speed: 'medium', vision: true },
        ],
        defaultModel: 'claude-haiku-4-5-20251001',
    },
    openrouter: {
        name: 'OpenRouter',
        baseUrl: 'https://openrouter.ai/api/v1/chat/completions',
        keyField: 'openrouterApiKey',
        models: [
            { id: 'google/gemma-4-31b-it:free', name: 'Gemma 4 31B (Free, Vision)', contextWindow: 262144, speed: 'fast', vision: true },
            { id: 'google/gemma-4-27b-a4b-it:free', name: 'Gemma 4 26B MoE (Free, Fast)', contextWindow: 262144, speed: 'fastest', vision: true },
            { id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6 (Paid)', contextWindow: 200000, speed: 'medium', vision: true },
            { id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5 (Paid)', contextWindow: 200000, speed: 'fastest', vision: true },
            { id: 'google/gemini-2.5-flash-preview', name: 'Gemini 2.5 Flash (Paid)', contextWindow: 1048576, speed: 'fast', vision: true },
            { id: 'openai/gpt-4.1-mini', name: 'GPT-4.1 Mini (Paid)', contextWindow: 1047576, speed: 'fast', vision: true },
        ],
        defaultModel: 'google/gemma-4-31b-it:free',
    },
    gemini: {
        name: 'Gemini (Google)',
        baseUrl: null,
        keyField: 'geminiApiKey',
        models: [
            { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash (Free)', contextWindow: 1048576, speed: 'fast', vision: true },
            { id: 'gemini-2.5-flash-lite-preview', name: 'Gemini 2.5 Flash Lite (Free)', contextWindow: 1048576, speed: 'fastest', vision: true },
        ],
        defaultModel: 'gemini-2.5-flash',
    },
};

// ── Timeout + Retry Configuration ──────────────────────────────────────────

const CLOUD_TIMEOUT_MS = 15000;
const MAX_RETRIES = 2;
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
    return errorMsg && /API (40[0-9]|4[1-9][0-9])/.test(errorMsg);
}

// ── OpenAI-Compatible Streaming (OpenRouter) ─────────────────────────────

async function streamOpenAICompatible({ baseUrl, apiKey, model, messages, onToken, onDone, onError, providerName }) {
    let lastError = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CLOUD_TIMEOUT_MS);

        try {
            const headers = {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            };

            if (providerName === 'OpenRouter') {
                headers['HTTP-Referer'] = 'https://github.com/AswaniSahoo/KAITE';
                headers['X-Title'] = 'KAITE';
            }

            const isFreeModel = model.includes(':free');
            const response = await fetch(baseUrl, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    model,
                    messages,
                    stream: true,
                    temperature: 0.7,
                    max_tokens: isFreeModel ? 4096 : 256,
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
            let leftover = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = leftover + decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                leftover = lines.pop() || '';

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
            const errorMsg = isTimeout ? `${providerName} timed out (${CLOUD_TIMEOUT_MS / 1000}s)` : error.message;

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

// ── Anthropic Streaming ──────────────────────────────────────────────────

async function streamAnthropic({ apiKey, model, messages, systemPrompt, onToken, onDone, onError }) {
    let lastError = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CLOUD_TIMEOUT_MS);

        try {
            const conversationMsgs = messages
                .filter(m => m.role !== 'system')
                .map(m => ({
                    role: m.role === 'assistant' ? 'assistant' : 'user',
                    content: m.content,
                }));

            const response = await fetch('https://api.anthropic.com/v1/messages', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': apiKey,
                    'anthropic-version': '2023-06-01',
                },
                body: JSON.stringify({
                    model,
                    max_tokens: 256,
                    system: systemPrompt || 'You are a helpful assistant.',
                    messages: conversationMsgs,
                    stream: true,
                }),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Anthropic API ${response.status}: ${errorText.substring(0, 200)}`);
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
                        const data = line.slice(6);
                        try {
                            const json = JSON.parse(data);
                            if (json.type === 'content_block_delta' && json.delta?.text) {
                                fullText += json.delta.text;
                                onToken(fullText, isFirst);
                                isFirst = false;
                            }
                        } catch (_parseError) {
                            // Skip invalid JSON chunks
                        }
                    }
                }
            }

            if (leftover.startsWith('data: ')) {
                try {
                    const json = JSON.parse(leftover.slice(6));
                    if (json.type === 'content_block_delta' && json.delta?.text) {
                        fullText += json.delta.text;
                    }
                } catch (_e) {
                    // ignore
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
            const errorMsg = isTimeout ? `Anthropic timed out (${CLOUD_TIMEOUT_MS / 1000}s)` : error.message;

            if (isNonRetryableError(error.message)) {
                console.error(`[Anthropic] Non-retryable: ${errorMsg.substring(0, 100)}`);
                break;
            }

            console.error(`[Anthropic] Attempt ${attempt}/${MAX_RETRIES} failed: ${errorMsg}`);

            if (attempt < MAX_RETRIES) {
                const delay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt - 1);
                console.log(`[Anthropic] Retrying in ${delay}ms...`);
                await sleep(delay);
            }
        }
    }

    const finalError = `Anthropic failed: ${lastError?.message || 'Unknown error'}`;
    console.error(finalError);
    onError(finalError);
    return null;
}

// ── Single Provider Dispatch ──────────────────────────────────────────────

async function callSingleProvider(provider, model, apiKey, messages, systemPrompt, onToken, onDone, onError) {
    const providerConfig = PROVIDERS[provider];
    if (!providerConfig) {
        onError(`Unknown provider: ${provider}`);
        return null;
    }

    if (provider === 'anthropic') {
        return streamAnthropic({ apiKey, model, messages, systemPrompt, onToken, onDone, onError });
    }

    if (provider === 'gemini') {
        const historyMsgs = messages.filter(m => m.role !== 'system');
        return streamGeminiSDK({ apiKey, model, messages: historyMsgs, systemPrompt, onToken, onDone, onError });
    }

    // OpenRouter (OpenAI-compatible)
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

function buildFailoverChain(primaryProvider, primaryModel, apiKeys) {
    const chain = [];

    const providerOrder = [
        { key: 'anthropic', model: PROVIDERS.anthropic.defaultModel },
        { key: 'openrouter', model: PROVIDERS.openrouter.defaultModel },
        { key: 'gemini', model: PROVIDERS.gemini.defaultModel },
    ];

    // Primary goes first with user-selected model
    if (primaryProvider) {
        const key = apiKeys[primaryProvider] || '';
        if (key) {
            chain.push({
                provider: primaryProvider,
                model: primaryModel || PROVIDERS[primaryProvider]?.defaultModel,
                apiKey: key,
            });
        }
    }

    // Add remaining providers as fallbacks
    for (const entry of providerOrder) {
        if (entry.key === primaryProvider) continue;

        const key = apiKeys[entry.key] || '';
        if (key) {
            chain.push({ provider: entry.key, model: entry.model, apiKey: key });
        }
    }

    return chain;
}

async function sendToProvider(transcription, opts) {
    const {
        provider: primaryProvider,
        model: primaryModel,
        apiKey: primaryApiKey,
        apiKeys,
        systemPrompt,
        conversationHistory,
        onToken,
        onDone,
        onError,
        onStatus,
    } = opts;

    if (!transcription || transcription.trim() === '') {
        console.log('Empty transcription, skipping');
        return { text: null, provider: null, model: null };
    }

    const allKeys = apiKeys || {
        openrouter: '',
        gemini: '',
        [primaryProvider]: primaryApiKey || '',
    };

    const chain = buildFailoverChain(primaryProvider, primaryModel, allKeys);

    if (chain.length === 0) {
        const msg = 'No API keys configured. Add an Anthropic, OpenRouter, or Gemini key in Settings.';
        onError(msg);
        return { text: null, provider: null, model: null };
    }

    const messages = [
        { role: 'system', content: systemPrompt || 'You are a helpful assistant.' },
        ...trimConversationHistory(conversationHistory, 42000),
        { role: 'user', content: transcription.trim() },
    ];

    for (let i = 0; i < chain.length; i++) {
        const { provider, model, apiKey } = chain[i];
        const isLast = i === chain.length - 1;

        const statusFn = onStatus || (() => {});
        statusFn(`Thinking (${PROVIDERS[provider]?.name || provider} / ${model})...`);

        console.log(`[Failover ${i + 1}/${chain.length}] Trying ${provider}/${model}...`);

        let succeeded = false;
        let resultText = null;

        await callSingleProvider(
            provider,
            model,
            apiKey,
            messages,
            systemPrompt,
            (displayText, isFirst) => {
                onToken(displayText, isFirst);
            },
            cleanedResponse => {
                succeeded = true;
                resultText = cleanedResponse;
            },
            errorMsg => {
                console.error(`[Failover] ${provider}/${model} failed: ${errorMsg}`);
                if (isLast) {
                    onError(`All providers failed. Last error: ${errorMsg}`);
                }
            }
        );

        if (succeeded && resultText) {
            console.log(`[Failover] Success with ${provider}/${model}`);
            onDone(resultText);
            return { text: resultText, provider, model };
        }

        if (!isLast) {
            console.log(`[Failover] Cascading to next provider...`);
            statusFn(`${PROVIDERS[provider]?.name} failed, trying next...`);
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
    MAX_RETRIES,
};
