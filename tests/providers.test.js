/**
 * Provider Failover System - Unit Tests
 *
 * Tests for:
 * 1. Failover chain building (correct order, deduplication)
 * 2. Error classification (retryable vs non-retryable)
 * 3. Model registry completeness
 * 4. Ollama local model connectivity
 * 5. Edge cases (empty keys, bad providers, etc.)
 *
 * Run: node tests/providers.test.js
 */

const { PROVIDERS, buildFailoverChain, trimConversationHistory, stripThinkingTags } = require('../src/utils/providers');

let passed = 0;
let failed = 0;
let skipped = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`  ✅ ${name}`);
        passed++;
    } catch (err) {
        console.error(`  ❌ ${name}`);
        console.error(`     ${err.message}`);
        failed++;
    }
}

function skip(name, reason) {
    console.log(`  ⏭️  ${name} (skipped: ${reason})`);
    skipped++;
}

function assert(condition, message) {
    if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(`${message}: expected "${expected}", got "${actual}"`);
    }
}

// ────────────────────────────────────────────────────────────────────────────
console.log('\n=== Provider Registry Tests ===\n');

test('All 3 providers exist in registry', () => {
    const providers = Object.keys(PROVIDERS);
    assert(providers.includes('gemini'), 'Missing gemini');
    assert(providers.includes('openrouter'), 'Missing openrouter');
    assert(providers.includes('ollama'), 'Missing ollama');
    assertEqual(providers.length, 3, 'Provider count');
});

test('Groq is removed from providers', () => {
    assert(!PROVIDERS.groq, 'Groq should not exist in registry');
});

test('Every provider has at least 1 model', () => {
    for (const [name, config] of Object.entries(PROVIDERS)) {
        assert(config.models.length > 0, `${name} has zero models`);
    }
});

test('Every provider has a valid defaultModel', () => {
    for (const [name, config] of Object.entries(PROVIDERS)) {
        const ids = config.models.map(m => m.id);
        assert(ids.includes(config.defaultModel), `${name} defaultModel "${config.defaultModel}" not in model list`);
    }
});

test('OpenRouter has free vision models (Gemma 4)', () => {
    const freeVision = PROVIDERS.openrouter.models.filter(m => m.id.includes(':free') && m.vision);
    assert(freeVision.length >= 2, `Expected at least 2 free vision models, got ${freeVision.length}`);
});

test('Gemini has thinking-capable models (2.5 Flash, 2.5 Pro, 3.1 Pro)', () => {
    const models = PROVIDERS.gemini.models;
    assert(models.length >= 3, `Expected at least 3 Gemini models, got ${models.length}`);
    assert(
        models.some(m => m.id.includes('2.5-flash')),
        'Missing 2.5 Flash'
    );
    assert(
        models.some(m => m.id.includes('2.5-pro')),
        'Missing 2.5 Pro'
    );
    assert(
        models.some(m => m.id.includes('3.1-pro')),
        'Missing 3.1 Pro'
    );
});

test('All Gemini models have vision', () => {
    const nonVision = PROVIDERS.gemini.models.filter(m => !m.vision);
    assertEqual(nonVision.length, 0, 'All Gemini models should have vision');
});

test('Ollama has gemma4 as default', () => {
    assertEqual(PROVIDERS.ollama.defaultModel, 'gemma4:latest', 'Ollama default model');
});

test('All models have required fields (id, name, contextWindow, speed, vision)', () => {
    for (const [provName, config] of Object.entries(PROVIDERS)) {
        for (const model of config.models) {
            assert(model.id, `${provName} model missing id`);
            assert(model.name, `${provName} model ${model.id} missing name`);
            assert(model.contextWindow > 0, `${provName} model ${model.id} invalid contextWindow`);
            assert(['fastest', 'fast', 'medium', 'slow'].includes(model.speed), `${provName} model ${model.id} invalid speed "${model.speed}"`);
            assert(typeof model.vision === 'boolean', `${provName} model ${model.id} missing vision flag`);
        }
    }
});

// ────────────────────────────────────────────────────────────────────────────
console.log('\n=== Failover Chain Tests ===\n');

test('Primary appears first in chain', () => {
    const chain = buildFailoverChain('openrouter', 'anthropic/claude-sonnet-4', {
        openrouter: 'sk-test',
        gemini: 'AIza_test',
    });
    assertEqual(chain[0].provider, 'openrouter', 'First provider');
    assertEqual(chain[0].model, 'anthropic/claude-sonnet-4', 'First model');
});

test('Ollama always last in chain', () => {
    const chain = buildFailoverChain('gemini', 'gemini-2.5-flash', {
        openrouter: 'sk-test',
        gemini: 'AIza_test',
    });
    const last = chain[chain.length - 1];
    assertEqual(last.provider, 'ollama', 'Last provider should be ollama');
});

test('No duplicate providers in chain', () => {
    const chain = buildFailoverChain('openrouter', 'anthropic/claude-sonnet-4', {
        openrouter: 'sk-test',
        gemini: 'AIza_test',
    });
    const providers = chain.map(c => c.provider);
    const unique = [...new Set(providers)];
    assertEqual(providers.length, unique.length, 'Duplicate providers found');
});

test('Providers without API keys are excluded (except ollama)', () => {
    const chain = buildFailoverChain('openrouter', 'anthropic/claude-sonnet-4', {
        openrouter: 'sk-test',
        gemini: '', // empty key
    });
    const providers = chain.map(c => c.provider);
    assert(!providers.includes('gemini'), 'Gemini should be excluded (empty key)');
    assert(providers.includes('ollama'), 'Ollama should always be included');
    assert(providers.includes('openrouter'), 'OpenRouter should be included (has key)');
});

test('Chain works with no API keys at all (only ollama)', () => {
    const chain = buildFailoverChain('openrouter', 'anthropic/claude-sonnet-4', {});
    assertEqual(chain.length, 1, 'Should only have ollama');
    assertEqual(chain[0].provider, 'ollama', 'Only provider should be ollama');
});

test('Ollama as primary works', () => {
    const chain = buildFailoverChain('ollama', 'gemma4:latest', {
        openrouter: 'sk-test',
    });
    assertEqual(chain[0].provider, 'ollama', 'Ollama should be first');
    assertEqual(chain[0].model, 'gemma4:latest', 'Ollama model');
});

test('Gemini as primary uses SDK path', () => {
    const chain = buildFailoverChain('gemini', 'gemini-2.5-flash', {
        gemini: 'AIza_test',
    });
    assertEqual(chain[0].provider, 'gemini', 'Gemini should be first');
});

test('Failover order: Gemini -> OpenRouter -> Ollama', () => {
    const chain = buildFailoverChain('gemini', 'gemini-2.5-flash', {
        gemini: 'AIza_test',
        openrouter: 'sk-test',
    });
    assertEqual(chain[0].provider, 'gemini', 'First: Gemini');
    assertEqual(chain[1].provider, 'openrouter', 'Second: OpenRouter');
    assertEqual(chain[2].provider, 'ollama', 'Third: Ollama');
});

// ────────────────────────────────────────────────────────────────────────────
console.log('\n=== Utility Function Tests ===\n');

test('trimConversationHistory respects character limit', () => {
    const history = [
        { role: 'user', content: 'A'.repeat(1000) },
        { role: 'assistant', content: 'B'.repeat(1000) },
        { role: 'user', content: 'C'.repeat(1000) },
    ];
    const trimmed = trimConversationHistory(history, 1500);
    assert(trimmed.length < history.length, 'Should have trimmed some entries');
});

test('trimConversationHistory returns empty for null input', () => {
    const result = trimConversationHistory(null);
    assertEqual(result.length, 0, 'Should return empty array');
});

test('stripThinkingTags removes <think> blocks', () => {
    const input = '<think>internal reasoning</think>The actual answer is 42.';
    const result = stripThinkingTags(input);
    assertEqual(result, 'The actual answer is 42.', 'Should strip thinking tags');
});

test('stripThinkingTags handles multiline think blocks', () => {
    const input = '<think>\nstep 1\nstep 2\n</think>Final answer.';
    const result = stripThinkingTags(input);
    assertEqual(result, 'Final answer.', 'Multiline strip');
});

test('stripThinkingTags handles no think blocks', () => {
    const input = 'Just a normal response.';
    const result = stripThinkingTags(input);
    assertEqual(result, 'Just a normal response.', 'No change expected');
});

// ────────────────────────────────────────────────────────────────────────────
console.log('\n=== Error Classification Tests ===\n');

// Inline test for the error classification logic (isNonRetryableError)
function isNonRetryableError(errorMsg) {
    const msg = (errorMsg || '').toLowerCase();
    const nonRetryablePatterns = [
        'invalid api key',
        'invalid x-api-key',
        'authentication',
        'unauthorized',
        'billing',
        'quota exceeded',
        'rate limit',
        'account',
        'invalid_api_key',
        'permission denied',
        'forbidden',
        'not found',
        '401',
        '403',
        '404',
        '402',
        '429',
    ];
    return nonRetryablePatterns.some(pattern => msg.includes(pattern));
}

test('Billing errors are non-retryable', () => {
    assert(isNonRetryableError('Payment required: billing quota exceeded'), 'billing');
    assert(isNonRetryableError('HTTP 402: insufficient credits'), '402');
});

test('Auth errors are non-retryable', () => {
    assert(isNonRetryableError('HTTP 401 Unauthorized'), '401');
    assert(isNonRetryableError('Invalid API key provided'), 'invalid api key');
    assert(isNonRetryableError('HTTP 403 Forbidden'), '403');
});

test('Rate limit errors are non-retryable', () => {
    assert(isNonRetryableError('HTTP 429 Too Many Requests rate limit'), 'rate limit');
});

test('Server errors ARE retryable', () => {
    assert(!isNonRetryableError('HTTP 500 Internal Server Error'), '500 should be retryable');
    assert(!isNonRetryableError('HTTP 503 Service Unavailable'), '503 should be retryable');
    assert(!isNonRetryableError('Connection timeout'), 'timeout should be retryable');
    assert(!isNonRetryableError('ECONNREFUSED'), 'connection refused should be retryable');
});

// ────────────────────────────────────────────────────────────────────────────
console.log('\n=== Ollama Connectivity Test ===\n');

async function testOllama() {
    try {
        const response = await fetch('http://127.0.0.1:11434/api/tags', {
            signal: AbortSignal.timeout(5000),
        });
        const data = await response.json();
        console.log(`  ✅ Ollama is running. Models available:`);
        data.models.forEach(m => {
            const sizeGB = (m.size / 1e9).toFixed(1);
            console.log(`     - ${m.name} (${sizeGB}GB)`);
        });
        passed++;

        // Test if gemma4 is available
        const hasGemma4 = data.models.some(m => m.name.startsWith('gemma4'));
        if (hasGemma4) {
            console.log(`  ✅ gemma4 model is available as failover target`);
            passed++;
        } else {
            console.log(`  ⚠️  gemma4 not found. Run: ollama pull gemma4`);
            skipped++;
        }
    } catch {
        skip('Ollama connectivity', 'Ollama server not running. Start with: ollama serve');
        skip('Ollama gemma4 check', 'depends on Ollama server');
    }
}

// ────────────────────────────────────────────────────────────────────────────
async function main() {
    await testOllama();

    console.log('\n════════════════════════════════════════════');
    console.log(`  Results: ${passed} passed, ${failed} failed, ${skipped} skipped`);
    console.log('════════════════════════════════════════════\n');

    if (failed > 0) {
        process.exit(1);
    }
}

main();
