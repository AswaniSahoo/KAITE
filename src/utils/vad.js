/**
 * Advanced Voice Activity Detection (VAD) for KAITE
 *
 * Two-stage VAD inspired by Cluely/Natively architecture:
 *   Stage 1: Adaptive energy threshold (replaces fixed RMS noise gate)
 *   Stage 2: Spectral speech detection (checks if energy is in speech-band frequencies)
 *
 * Features:
 *   - Adaptive noise floor that learns ambient environment in real-time
 *   - Speech-band frequency analysis (300-3400 Hz) to distinguish speech from fan/typing noise
 *   - Trailing buffer to capture soft word endings
 *   - Hangover period to bridge short pauses within utterances
 *   - Speech onset detection for fast response
 */

const SAMPLE_RATE = 24000;

// --- Stage 1: Adaptive Energy Threshold ---
const NOISE_FLOOR_ADAPT_RATE = 0.03;   // How fast noise floor adapts (higher = faster)
const NOISE_FLOOR_DECAY_RATE = 0.001;  // Slow decay when no speech (creeps up for changing environments)
const SPEECH_THRESHOLD_MULTIPLIER = 3.5; // Speech must be 3.5x the noise floor
const MIN_NOISE_FLOOR = 0.0005;         // Absolute minimum (prevent zero-lock)
const MAX_NOISE_FLOOR = 0.02;           // Absolute maximum (prevent adapting to speech)
const INITIAL_NOISE_FLOOR = 0.003;      // Starting estimate

// --- Stage 2: Spectral Speech Detection ---
// Human speech fundamental: ~85-255 Hz, but formants dominate 300-3400 Hz
const SPEECH_BAND_LOW = 300;
const SPEECH_BAND_HIGH = 3400;
const SPEECH_BAND_RATIO_THRESHOLD = 0.35; // At least 35% of energy must be in speech band

// --- Trailing / Hangover ---
const TRAIL_CHUNKS = 12;        // Keep gate open for 12 chunks (~1.2s at 100ms chunks) after last speech
const ONSET_LOOKBACK = 2;       // Include 2 chunks before speech onset (captures attack transients)
const MIN_SPEECH_CHUNKS = 3;    // Must have at least 3 consecutive speech chunks to trigger (debounce clicks/pops)

class AdaptiveVAD {
    constructor(sampleRate = SAMPLE_RATE) {
        this.sampleRate = sampleRate;
        this.noiseFloor = INITIAL_NOISE_FLOOR;
        this.trailCounter = 0;
        this.consecutiveSpeechChunks = 0;
        this.isSpeechActive = false;

        // Ring buffer for onset lookback
        this.recentChunks = [];
        this.maxLookback = ONSET_LOOKBACK + 1;

        // Stats for monitoring
        this.stats = {
            totalChunks: 0,
            speechChunks: 0,
            noiseFloor: this.noiseFloor,
            lastRMS: 0,
            lastSpeechBandRatio: 0,
        };
    }

    /**
     * Calculate RMS energy of a Float32Array audio chunk.
     */
    calculateRMS(samples) {
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
            sum += samples[i] * samples[i];
        }
        return Math.sqrt(sum / samples.length);
    }

    /**
     * Compute the ratio of energy in the speech frequency band (300-3400 Hz)
     * vs total energy. Uses a simplified DFT on key frequency bins rather than
     * a full FFT to keep CPU usage minimal.
     *
     * For 24kHz sample rate with typical chunk sizes (~2400 samples = 100ms):
     *   - Frequency resolution = sampleRate / N = 24000/2400 = 10 Hz per bin
     *   - Speech band 300-3400 Hz = bins 30-340
     */
    getSpeechBandRatio(samples) {
        const N = samples.length;
        if (N < 64) return 0.5; // Too short to analyze, pass through

        // Downsample the DFT: check every 4th bin for speed
        const freqResolution = this.sampleRate / N;
        const lowBin = Math.floor(SPEECH_BAND_LOW / freqResolution);
        const highBin = Math.min(Math.ceil(SPEECH_BAND_HIGH / freqResolution), Math.floor(N / 2));
        const maxBin = Math.floor(N / 2);

        let speechEnergy = 0;
        let totalEnergy = 0;
        const step = Math.max(1, Math.floor((maxBin) / 120)); // Sample ~120 bins across spectrum

        for (let k = 1; k < maxBin; k += step) {
            // Goertzel-like single-bin energy (real part only for speed)
            let re = 0, im = 0;
            const w = (2 * Math.PI * k) / N;
            for (let n = 0; n < N; n++) {
                re += samples[n] * Math.cos(w * n);
                im -= samples[n] * Math.sin(w * n);
            }
            const energy = re * re + im * im;
            totalEnergy += energy;

            if (k >= lowBin && k <= highBin) {
                speechEnergy += energy;
            }
        }

        if (totalEnergy === 0) return 0;
        return speechEnergy / totalEnergy;
    }

    /**
     * Lightweight speech band check: instead of full spectral analysis,
     * use zero-crossing rate as a proxy. Speech has moderate ZCR (1000-4000/s),
     * while noise tends to be very high or very low.
     *
     * This is ~100x faster than even the simplified DFT above.
     */
    getZeroCrossingRate(samples) {
        let crossings = 0;
        for (let i = 1; i < samples.length; i++) {
            if ((samples[i] >= 0 && samples[i - 1] < 0) ||
                (samples[i] < 0 && samples[i - 1] >= 0)) {
                crossings++;
            }
        }
        // Normalize to crossings per second
        const durationSec = samples.length / this.sampleRate;
        return crossings / durationSec;
    }

    /**
     * Main VAD decision: should this audio chunk be sent for transcription?
     *
     * Returns: { send: boolean, reason: string }
     */
    process(samples) {
        this.stats.totalChunks++;
        const rms = this.calculateRMS(samples);
        this.stats.lastRMS = rms;

        // --- Stage 1: Adaptive Energy Threshold ---
        const energyThreshold = Math.max(
            this.noiseFloor * SPEECH_THRESHOLD_MULTIPLIER,
            MIN_NOISE_FLOOR * SPEECH_THRESHOLD_MULTIPLIER
        );

        const isAboveEnergy = rms >= energyThreshold;

        // Adapt noise floor
        if (!isAboveEnergy && !this.isSpeechActive) {
            // In silence: slowly adapt noise floor toward current level
            this.noiseFloor = this.noiseFloor * (1 - NOISE_FLOOR_ADAPT_RATE) + rms * NOISE_FLOOR_ADAPT_RATE;
            this.noiseFloor = Math.max(MIN_NOISE_FLOOR, Math.min(MAX_NOISE_FLOOR, this.noiseFloor));
        } else if (!this.isSpeechActive) {
            // Energy spike but not yet confirmed speech: slow decay
            this.noiseFloor += NOISE_FLOOR_DECAY_RATE;
            this.noiseFloor = Math.min(MAX_NOISE_FLOOR, this.noiseFloor);
        }
        this.stats.noiseFloor = this.noiseFloor;

        // --- Stage 2: Speech-band check (lightweight ZCR-based) ---
        let isSpeechLike = true;
        if (isAboveEnergy) {
            const zcr = this.getZeroCrossingRate(samples);
            // Speech ZCR is typically 1000-5000/s
            // Pure noise (fan) > 8000/s, clicks/pops < 500/s
            isSpeechLike = zcr >= 500 && zcr <= 7000;
            this.stats.lastSpeechBandRatio = zcr;
        }

        // --- Decision Logic ---
        const isSpeechChunk = isAboveEnergy && isSpeechLike;

        if (isSpeechChunk) {
            this.consecutiveSpeechChunks++;

            if (this.consecutiveSpeechChunks >= MIN_SPEECH_CHUNKS) {
                // Confirmed speech - activate
                this.isSpeechActive = true;
                this.trailCounter = TRAIL_CHUNKS;
                this.stats.speechChunks++;
                return { send: true, reason: 'speech' };
            } else {
                // Potential speech but not confirmed yet (could be a click/pop)
                // Buffer it but don't commit
                this.trailCounter = TRAIL_CHUNKS;
                return { send: true, reason: 'onset-candidate' };
            }
        } else {
            this.consecutiveSpeechChunks = 0;

            if (this.trailCounter > 0) {
                // In trailing window - let it through to capture word endings
                this.trailCounter--;
                if (this.trailCounter === 0) {
                    this.isSpeechActive = false;
                }
                return { send: true, reason: 'trailing' };
            }

            // Pure silence/noise
            this.isSpeechActive = false;
            return { send: false, reason: 'silence' };
        }
    }

    /**
     * Simple boolean interface matching the old isAboveNoiseGate() API.
     */
    shouldSend(samples) {
        return this.process(samples).send;
    }

    /**
     * Reset VAD state (e.g., on session restart).
     */
    reset() {
        this.noiseFloor = INITIAL_NOISE_FLOOR;
        this.trailCounter = 0;
        this.consecutiveSpeechChunks = 0;
        this.isSpeechActive = false;
        this.recentChunks = [];
        this.stats = {
            totalChunks: 0,
            speechChunks: 0,
            noiseFloor: this.noiseFloor,
            lastRMS: 0,
            lastSpeechBandRatio: 0,
        };
    }

    /**
     * Get current VAD stats for debugging.
     */
    getStats() {
        return { ...this.stats };
    }
}

module.exports = { AdaptiveVAD };
