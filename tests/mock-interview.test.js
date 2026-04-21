/**
 * Mock Interview Test
 *
 * Sends real interview questions to the configured text provider and measures:
 * - Response time (time to first token + total)
 * - Response length (words, sentences)
 * - Response quality (no clarifying questions, English only, concise)
 *
 * Usage: node tests/mock-interview.test.js [provider]
 *   provider: anthropic | openrouter | gemini | ollamaCloud | ollama
 *   If omitted, auto-detects the first available provider.
 *
 * Requires .env with API keys configured.
 */

// Load env using the project's own loader (no dotenv dependency)
require('../src/storage');

const { PROVIDERS, sendToProvider } = require('../src/utils/providers');
const { getSystemPrompt } = require('../src/utils/prompts');

// ── Test Configuration ─────────────────────────────────────────────────────
const MOCK_CV_CONTEXT = `
Name: Akash Kar
Education: B.Tech Biomedical Engineering, NIT Rourkela (2022-2026)
Projects:
- House Price Prediction: End-to-end ML pipeline, 80+ features, XGBoost + Random Forest ensemble, R² 0.9168, MAE 0.1082
- AI Fitness Telegram Bot: OpenAI LLM API, Aiogram framework, BMI/calorie calculators, deployed on Railway
- Hybrid Music Recommendation System: Content-based + collaborative filtering, Spotify API
Skills: Python, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, FastAPI, React, Node.js
Experience: Biosignal Laboratory - ECG preprocessing, cardiac anomaly classification
`;

const INTERVIEW_QUESTIONS = [
    'Tell me about yourself.',
    'What is the difference between R squared and adjusted R squared?',
    'Explain your house price prediction project.',
    'What challenges did you face in your Telegram bot project?',
    'Why should we hire you?',
];

// ── API Key Detection ──────────────────────────────────────────────────────
function getApiKeys() {
    return {
        anthropic: process.env.ANTHROPIC_API_KEY || '',
        openrouter: process.env.OPENROUTER_API_KEY || '',
        gemini: process.env.GEMINI_API_KEY || '',
        ollamaCloud: process.env.OLLAMA_CLOUD_API_KEY || '',
    };
}

function detectProvider(requestedProvider) {
    const keys = getApiKeys();
    if (requestedProvider && (requestedProvider === 'ollama' || keys[requestedProvider])) {
        return requestedProvider;
    }
    const chain = ['anthropic', 'openrouter', 'gemini', 'ollamaCloud', 'ollama'];
    return chain.find((p) => p === 'ollama' || keys[p]) || 'ollama';
}

// ── Quality Checks ─────────────────────────────────────────────────────────
function checkResponseQuality(response) {
    const issues = [];

    const clarifyPatterns = [
        /could you (tell|clarify|explain)/i,
        /what (industry|role|company)/i,
        /can you (provide|share|tell)/i,
        /I don't have enough context/i,
        /I need more information/i,
        /what do you mean/i,
    ];
    for (const pattern of clarifyPatterns) {
        if (pattern.test(response)) {
            issues.push(`❌ Contains clarifying question: "${response.match(pattern)[0]}"`);
        }
    }

    const latinChars = (response.match(/[a-zA-Z0-9\s.,!?'";\-:()]/g) || []).length;
    const totalChars = response.replace(/\s/g, '').length;
    if (totalChars > 0 && latinChars / totalChars < 0.8) {
        issues.push('❌ Response may not be in English');
    }

    const wordCount = response.split(/\s+/).length;
    if (wordCount > 150) {
        issues.push(`⚠️ Response too long: ${wordCount} words (target: under 100)`);
    }

    const fluffPatterns = [/^(great|good|excellent|wonderful) question/i, /^I think you're asking/i, /^That's a (great|good)/i];
    for (const pattern of fluffPatterns) {
        if (pattern.test(response.trim())) {
            issues.push(`⚠️ Starts with fluff: "${response.trim().substring(0, 40)}..."`);
        }
    }

    return issues;
}

// ── Main Test ──────────────────────────────────────────────────────────────
async function runMockInterview() {
    const requestedProvider = process.argv[2] || null;
    const provider = detectProvider(requestedProvider);
    const keys = getApiKeys();
    const model = PROVIDERS[provider]?.defaultModel || 'unknown';

    console.log('\n' + '═'.repeat(70));
    console.log('  MOCK INTERVIEW TEST');
    console.log('═'.repeat(70));
    console.log(`  Provider : ${PROVIDERS[provider]?.name || provider}`);
    console.log(`  Model    : ${model}`);
    console.log(`  Questions: ${INTERVIEW_QUESTIONS.length}`);
    console.log('═'.repeat(70) + '\n');

    const systemPrompt = getSystemPrompt('interview', '', false, MOCK_CV_CONTEXT);
    let conversationHistory = [];
    const results = [];
    let totalTime = 0;
    let passedQuality = 0;

    for (let i = 0; i < INTERVIEW_QUESTIONS.length; i++) {
        const question = INTERVIEW_QUESTIONS[i];
        console.log(`─── Q${i + 1}: "${question}" ───`);

        const startTime = Date.now();
        let firstTokenTime = 0;
        let fullResponse = '';

        try {
            await sendToProvider(question, {
                provider,
                model,
                apiKey: keys[provider] || '',
                apiKeys: keys,
                systemPrompt,
                conversationHistory,
                ollamaHost: 'http://localhost:11434',
                onToken: (text, isFirst) => {
                    if (isFirst) {
                        firstTokenTime = Date.now() - startTime;
                    }
                    fullResponse = text;
                },
                onDone: (cleaned) => {
                    if (cleaned) {
                        fullResponse = cleaned;
                        conversationHistory.push({ role: 'user', content: question });
                        conversationHistory.push({ role: 'assistant', content: cleaned });
                    }
                },
                onError: (err) => {
                    console.error(`  ❌ Error: ${err}`);
                },
                onStatus: () => {},
            });
        } catch (err) {
            console.error(`  ❌ Exception: ${err.message}`);
            results.push({ question, error: err.message });
            continue;
        }

        const totalResponseTime = Date.now() - startTime;
        totalTime += totalResponseTime;
        const wordCount = fullResponse.split(/\s+/).length;
        const qualityIssues = checkResponseQuality(fullResponse);

        if (qualityIssues.length === 0) passedQuality++;

        const truncated = fullResponse.length > 200 ? fullResponse.substring(0, 200) + '...' : fullResponse;
        console.log(`  Response : ${truncated}`);
        console.log(`  TTFT     : ${firstTokenTime}ms`);
        console.log(`  Total    : ${(totalResponseTime / 1000).toFixed(1)}s`);
        console.log(`  Words    : ${wordCount}`);

        if (qualityIssues.length > 0) {
            qualityIssues.forEach((issue) => console.log(`  ${issue}`));
        } else {
            console.log('  ✅ Quality: PASS');
        }
        console.log();

        results.push({
            question,
            ttft: firstTokenTime,
            total: totalResponseTime,
            words: wordCount,
            qualityIssues: qualityIssues.length,
        });
    }

    // ── Summary ────────────────────────────────────────────────────────────
    const successResults = results.filter((r) => !r.error);
    const avgTTFT =
        successResults.length > 0 ? Math.round(successResults.reduce((s, r) => s + r.ttft, 0) / successResults.length) : 0;
    const avgTotal =
        successResults.length > 0
            ? (successResults.reduce((s, r) => s + r.total, 0) / successResults.length / 1000).toFixed(1)
            : 0;
    const avgWords =
        successResults.length > 0 ? Math.round(successResults.reduce((s, r) => s + r.words, 0) / successResults.length) : 0;

    console.log('═'.repeat(70));
    console.log('  RESULTS SUMMARY');
    console.log('═'.repeat(70));
    console.log(`  Provider       : ${PROVIDERS[provider]?.name} (${model})`);
    console.log(`  Questions      : ${INTERVIEW_QUESTIONS.length}`);
    console.log(`  Successful     : ${successResults.length}/${INTERVIEW_QUESTIONS.length}`);
    console.log(`  Avg TTFT       : ${avgTTFT}ms`);
    console.log(`  Avg Total Time : ${avgTotal}s`);
    console.log(`  Avg Words      : ${avgWords}`);
    console.log(`  Quality Pass   : ${passedQuality}/${successResults.length}`);
    console.log(`  Total Time     : ${(totalTime / 1000).toFixed(1)}s`);
    console.log('═'.repeat(70));

    const allPassed = successResults.length === INTERVIEW_QUESTIONS.length && passedQuality === successResults.length;
    if (avgTotal <= 3.0 && allPassed) {
        console.log('  🟢 VERDICT: READY FOR PRODUCTION');
    } else if (avgTotal <= 5.0) {
        console.log('  🟡 VERDICT: ACCEPTABLE (could be faster)');
    } else {
        console.log('  🔴 VERDICT: TOO SLOW FOR LIVE INTERVIEW');
    }
    console.log('═'.repeat(70) + '\n');
}

runMockInterview().catch(console.error);
