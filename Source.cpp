// Source_godlike_mt.cpp
// Godlike multithreaded CPU-grade voice processor for FM/AM/SSB over UHF radios
// Native Windows (WASAPI) realtime: default mic -> DSP -> default render
// - 3-thread pipeline: Capture -> DSP -> Render (uses multiple cores safely)
// - Heavy, "broadcast" chain with true-peak safety limiting to help avoid over-mod
// - Live parameter editing + save/load INI
//
// Build (VS2019/VS2022):
//  - C++20
//  - Link: ole32.lib uuid.lib avrt.lib
//
// Notes:
//  - You still MUST set your radio/SDR TX gain so that this app's "TP Ceiling" maps to safe deviation.
//    Press 'T' to output a calibration 1kHz tone @ -12 dBFS (5 seconds) and set TX chain accordingly.
//  - This code avoids 3rd party libs; everything is self-contained.

#define NOMINMAX
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <avrt.h>
#include <functiondiscoverykeys_devpkey.h>

#include <mmreg.h>
#include <ks.h>
#include <ksmedia.h>

#include <wrl/client.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>
#include <algorithm>
#include <string>
#include <conio.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include <condition_variable>

using Microsoft::WRL::ComPtr;

static constexpr float kPi = 3.14159265358979323846f;

template <typename T>
static inline T Clamp(T v, T lo, T hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

static inline float dbToLin(float db) { return std::pow(10.0f, db / 20.0f); }
static inline float linToDb(float x) { return 20.0f * std::log10(std::max(x, 1e-12f)); }

static std::atomic<bool> g_running{ true };

static BOOL WINAPI CtrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_CLOSE_EVENT || type == CTRL_BREAK_EVENT) {
        g_running.store(false);
        return TRUE;
    }
    return FALSE;
}

static void Die(const char* msg, HRESULT hr) {
    std::fprintf(stderr, "%s (hr=0x%08lx)\n", msg, (unsigned long)hr);
    std::exit(1);
}

static std::wstring GetDeviceName(IMMDevice* dev) {
    ComPtr<IPropertyStore> store;
    HRESULT hr = dev->OpenPropertyStore(STGM_READ, &store);
    if (FAILED(hr)) return L"(unknown)";

    PROPVARIANT var; PropVariantInit(&var);
    hr = store->GetValue(PKEY_Device_FriendlyName, &var);
    if (FAILED(hr)) { PropVariantClear(&var); return L"(unknown)"; }

    std::wstring name = var.pwszVal ? var.pwszVal : L"(unknown)";
    PropVariantClear(&var);
    return name;
}

static bool IsFloatFormat(const WAVEFORMATEX* wf) {
    if (!wf) return false;
    if (wf->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) return true;
    if (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        auto* wfx = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wf);
        return (wfx->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);
    }
    return false;
}

static bool IsPcm16(const WAVEFORMATEX* wf) {
    if (!wf) return false;
    if (wf->wFormatTag == WAVE_FORMAT_PCM && wf->wBitsPerSample == 16) return true;
    if (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        auto* wfx = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wf);
        return (wfx->SubFormat == KSDATAFORMAT_SUBTYPE_PCM && wf->wBitsPerSample == 16);
    }
    return false;
}

static ComPtr<IMMDevice> GetDefault(IMMDeviceEnumerator* en, EDataFlow flow, ERole role) {
    ComPtr<IMMDevice> dev;
    HRESULT hr = en->GetDefaultAudioEndpoint(flow, role, &dev);
    if (FAILED(hr)) return nullptr;
    return dev;
}

// --------------------------- SPSC Ring Buffer (float) ---------------------------
// Lock-free single-producer/single-consumer ring for realtime audio.
// Drops oldest samples if full (better than blocking audio threads).

struct SpscRing {
    std::vector<float> buf;
    const size_t cap = 0;
    std::atomic<size_t> r{ 0 }, w{ 0 };

    std::atomic<uint64_t> dropped{ 0 };

    explicit SpscRing(size_t capacity)
        : buf(capacity + 1), cap(capacity + 1) {}

    size_t size() const {
        size_t rr = r.load(std::memory_order_acquire);
        size_t ww = w.load(std::memory_order_acquire);
        if (ww >= rr) return ww - rr;
        return (cap - rr) + ww;
    }

    size_t freeSpace() const {
        return (cap - 1) - size();
    }


    uint64_t droppedCount() const { return dropped.load(std::memory_order_relaxed); }

    // Push samples; if insufficient space, drop oldest to make room.
    void push(const float* x, size_t n) {
        if (n == 0) return;
        size_t need = n;
        size_t fs = freeSpace();
        if (need > fs) {
            size_t drop = need - fs;
            size_t rr = r.load(std::memory_order_relaxed);
            rr = (rr + drop) % cap;
            r.store(rr, std::memory_order_release);
            dropped.fetch_add((uint64_t)drop, std::memory_order_relaxed);
        }

        size_t ww = w.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; i++) {
            buf[ww] = x[i];
            ww++;
            if (ww == cap) ww = 0;
        }
        w.store(ww, std::memory_order_release);
    }

    // Pop up to n samples, returns count popped
    size_t pop(float* out, size_t n) {
        size_t rr = r.load(std::memory_order_relaxed);
        size_t ww = w.load(std::memory_order_acquire);
        size_t avail = (ww >= rr) ? (ww - rr) : ((cap - rr) + ww);
        size_t take = std::min(n, avail);

        for (size_t i = 0; i < take; i++) {
            out[i] = buf[rr];
            rr++;
            if (rr == cap) rr = 0;
        }
        r.store(rr, std::memory_order_release);
        return take;
    }
};

// --------------------------- DSP building blocks ---------------------------

// RBJ biquad (Direct Form II Transposed)
struct Biquad {
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float z1 = 0, z2 = 0;

    void reset() { z1 = z2 = 0; }

    float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

    static Biquad LowPass(float fs, float fc, float Q) {
        Biquad q;
        fc = std::max(10.0f, std::min(fc, fs * 0.49f));
        float w0 = 2.0f * kPi * (fc / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);

        float bb0 = (1 - cw) * 0.5f;
        float bb1 = (1 - cw);
        float bb2 = (1 - cw) * 0.5f;
        float a0 = 1 + alpha;
        float aa1 = -2 * cw;
        float aa2 = 1 - alpha;

        q.b0 = bb0 / a0; q.b1 = bb1 / a0; q.b2 = bb2 / a0;
        q.a1 = aa1 / a0; q.a2 = aa2 / a0;
        return q;
    }

    static Biquad HighPass(float fs, float fc, float Q) {
        Biquad q;
        fc = std::max(10.0f, std::min(fc, fs * 0.49f));
        float w0 = 2.0f * kPi * (fc / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);

        float bb0 = (1 + cw) * 0.5f;
        float bb1 = -(1 + cw);
        float bb2 = (1 + cw) * 0.5f;
        float a0 = 1 + alpha;
        float aa1 = -2 * cw;
        float aa2 = 1 - alpha;

        q.b0 = bb0 / a0; q.b1 = bb1 / a0; q.b2 = bb2 / a0;
        q.a1 = aa1 / a0; q.a2 = aa2 / a0;
        return q;
    }

    static Biquad Peaking(float fs, float f0, float Q, float gainDb) {
        Biquad q;
        f0 = std::max(20.0f, std::min(f0, fs * 0.49f));
        float A = std::pow(10.0f, gainDb / 40.0f);
        float w0 = 2.0f * kPi * (f0 / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);

        float bb0 = 1 + alpha * A;
        float bb1 = -2 * cw;
        float bb2 = 1 - alpha * A;
        float a0 = 1 + alpha / A;
        float aa1 = -2 * cw;
        float aa2 = 1 - alpha / A;

        q.b0 = bb0 / a0; q.b1 = bb1 / a0; q.b2 = bb2 / a0;
        q.a1 = aa1 / a0; q.a2 = aa2 / a0;
        return q;
    }

    static Biquad LowShelf(float fs, float f0, float Q, float gainDb) {
        Biquad q;
        f0 = std::max(20.0f, std::min(f0, fs * 0.49f));
        float A = std::pow(10.0f, gainDb / 40.0f);
        float w0 = 2.0f * kPi * (f0 / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);
        float beta = std::sqrt(A) / Q;

        float bb0 = A * ((A + 1) - (A - 1) * cw + beta * sw);
        float bb1 = 2 * A * ((A - 1) - (A + 1) * cw);
        float bb2 = A * ((A + 1) - (A - 1) * cw - beta * sw);
        float a0 = (A + 1) + (A - 1) * cw + beta * sw;
        float aa1 = -2 * ((A - 1) + (A + 1) * cw);
        float aa2 = (A + 1) + (A - 1) * cw - beta * sw;

        q.b0 = bb0 / a0; q.b1 = bb1 / a0; q.b2 = bb2 / a0;
        q.a1 = aa1 / a0; q.a2 = aa2 / a0;
        return q;
    }

    static Biquad HighShelf(float fs, float f0, float Q, float gainDb) {
        Biquad q;
        f0 = std::max(20.0f, std::min(f0, fs * 0.49f));
        float A = std::pow(10.0f, gainDb / 40.0f);
        float w0 = 2.0f * kPi * (f0 / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);
        float beta = std::sqrt(A) / Q;

        float bb0 = A * ((A + 1) + (A - 1) * cw + beta * sw);
        float bb1 = -2 * A * ((A - 1) + (A + 1) * cw);
        float bb2 = A * ((A + 1) + (A - 1) * cw - beta * sw);
        float a0 = (A + 1) - (A - 1) * cw + beta * sw;
        float aa1 = 2 * ((A - 1) - (A + 1) * cw);
        float aa2 = (A + 1) - (A - 1) * cw - beta * sw;

        q.b0 = bb0 / a0; q.b1 = bb1 / a0; q.b2 = bb2 / a0;
        q.a1 = aa1 / a0; q.a2 = aa2 / a0;
        return q;
    }
};

// Update coefficients but preserve DF2 state
static inline void UpdateCoeffsPreserveState(Biquad& dst, const Biquad& src) {
    float z1 = dst.z1, z2 = dst.z2;
    dst = src;
    dst.z1 = z1; dst.z2 = z2;
}

// Envelope follower (exp smoothing on abs signal)
struct EnvFollower {
    float aA = 0, aR = 0;
    float env = 0;

    void set(float fs, float attackMs, float releaseMs) {
        aA = std::exp(-1.0f / (fs * (attackMs / 1000.0f)));
        aR = std::exp(-1.0f / (fs * (releaseMs / 1000.0f)));
    }
    float process(float xAbs) {
        if (xAbs > env) env = aA * env + (1.0f - aA) * xAbs;
        else            env = aR * env + (1.0f - aR) * xAbs;
        return env;
    }
};

// Gate/expander with hysteresis + hold
struct SmartGate {
    float fs = 48000.0f;
    bool enabled = true;

    float openThreshDb = -55.0f;
    float closeThreshDb = -60.0f;
    float ratio = 4.0f;
    float floorDb = -35.0f;
    float holdMs = 60.0f;

    EnvFollower env;
    EnvFollower gainSmooth;
    bool isOpen = true;
    int holdSamples = 0;
    int holdCount = 0;

    void init(float Fs) {
        fs = Fs;
        env.set(fs, 3.0f, 150.0f);
        gainSmooth.set(fs, 3.0f, 120.0f);
        env.env = 0; gainSmooth.env = 1.0f;
        isOpen = true;
        holdSamples = std::max(0, (int)std::round(fs * (holdMs / 1000.0f)));
        holdCount = holdSamples;
    }

    float process(float x) {
        if (!enabled) return x;

        // If holdMs is edited live, update holdSamples without resetting envelope.
        int hs = std::max(0, (int)std::round(fs * (holdMs / 1000.0f)));
        if (hs != holdSamples) {
            holdSamples = hs;
            holdCount = std::min(holdCount, holdSamples);
        }

        float e = env.process(std::fabs(x));
        float eDb = linToDb(e);

        if (isOpen) {
            if (eDb < closeThreshDb) {
                if (holdCount <= 0) isOpen = false;
                else holdCount--;
            }
            else {
                holdCount = holdSamples;
            }
        }
        else {
            if (eDb > openThreshDb) {
                isOpen = true;
                holdCount = holdSamples;
            }
        }

        float gainDb = 0.0f;
        if (!isOpen) {
            float under = openThreshDb - eDb;
            float att = under * (std::max(ratio, 1.0f) - 1.0f);
            float maxAtt = std::max(0.0f, -floorDb);
            gainDb = -std::min(att, maxAtt);
        }

        float target = dbToLin(gainDb);
        float smooth = gainSmooth.process(target);
        return x * smooth;
    }
};

// Slow RMS-ish leveler AGC
struct LevelerAGC {
    float fs = 48000.0f;
    bool enabled = true;

    float targetDb = -18.0f;
    float maxGainDb = 18.0f;
    float minGainDb = -12.0f;
    float attackMs = 35.0f;
    float releaseMs = 250.0f;

    EnvFollower env;
    float g = 1.0f;

    void init(float Fs) {
        fs = Fs;
        env.set(fs, 30.0f, 300.0f);
        env.env = 0;
        g = 1.0f;
    }

    float process(float x) {
        if (!enabled) return x;

        float e = env.process(std::fabs(x));
        float eDb = linToDb(e);

        float needDb = targetDb - eDb;
        needDb = Clamp(needDb, minGainDb, maxGainDb);
        float target = dbToLin(needDb);

        float aA = std::exp(-1.0f / (fs * (attackMs / 1000.0f)));
        float aR = std::exp(-1.0f / (fs * (releaseMs / 1000.0f)));

        if (target < g) g = aA * g + (1.0f - aA) * target;
        else            g = aR * g + (1.0f - aR) * target;

        return x * g;
    }
};

// Compressor
struct Compressor {
    float fs = 48000.0f;
    float threshDb = -24.0f;
    float ratio = 6.0f;
    float makeupDb = 0.0f;
    float makeup = 1.0f;
    bool enabled = true;

    float envAttackMs = 4.0f;
    float envReleaseMs = 120.0f;
    float gAttackMs = 2.0f;
    float gReleaseMs = 80.0f;

    EnvFollower env;
    EnvFollower gainSmooth;

    void init(float Fs) {
        fs = Fs;
        env.set(fs, envAttackMs, envReleaseMs);
        gainSmooth.set(fs, gAttackMs, gReleaseMs);
        env.env = 0; gainSmooth.env = 1.0f;
        setMakeupDb(makeupDb);
    }

    void setMakeupDb(float db) { makeupDb = db; makeup = dbToLin(db); }

    float process(float x) {
        if (!enabled) return x;
        float e = env.process(std::fabs(x));
        float eDb = linToDb(e);
        float over = eDb - threshDb;

        float grDb = 0.0f;
        if (over > 0.0f) {
            float r = std::max(ratio, 1.0f);
            grDb = -(over - (over / r));
        }

        float targetGain = dbToLin(grDb);
        float smooth = gainSmooth.process(targetGain);
        return x * smooth * makeup;
    }
};

// De-esser
struct DeEsser {
    float fs = 48000.0f;
    bool enabled = true;

    float detAmount = 0.6f;
    float threshDb = -30.0f;
    float maxRedDb = 12.0f;

    Biquad hp1, hp2, lp1, lp2;
    EnvFollower env;

    void init(float Fs) {
        fs = Fs;
        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        hp1 = Biquad::HighPass(fs, 4500.0f, Q1);
        hp2 = Biquad::HighPass(fs, 4500.0f, Q2);
        lp1 = Biquad::LowPass(fs, 8000.0f, Q1);
        lp2 = Biquad::LowPass(fs, 8000.0f, Q2);
        hp1.reset(); hp2.reset(); lp1.reset(); lp2.reset();
        env.set(fs, 2.0f, 90.0f);
        env.env = 0;
    }

    void updateBand(float lo, float hi) {
        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        UpdateCoeffsPreserveState(hp1, Biquad::HighPass(fs, lo, Q1));
        UpdateCoeffsPreserveState(hp2, Biquad::HighPass(fs, lo, Q2));
        UpdateCoeffsPreserveState(lp1, Biquad::LowPass(fs, hi, Q1));
        UpdateCoeffsPreserveState(lp2, Biquad::LowPass(fs, hi, Q2));
    }

    float process(float x) {
        if (!enabled) return x;

        float s = hp2.process(hp1.process(x));
        s = lp2.process(lp1.process(s));
        float e = env.process(std::fabs(s));
        float eDb = linToDb(e);

        float redDb = 0.0f;
        if (eDb > threshDb) {
            float over = eDb - threshDb;
            redDb = std::min(over * 1.6f, maxRedDb);
        }

        float hfGain = dbToLin(-redDb * detAmount);
        return x - (1.0f - hfGain) * s;
    }
};

// Exciter
struct Exciter {
    float fs = 48000.0f;
    bool enabled = true;
    float amount = 0.30f;
    float drive = 2.0f;
    float loHz = 1200.0f;
    float hiHz = 3400.0f;

    Biquad hp1, hp2, lp1, lp2;

    void init(float Fs) {
        fs = Fs;
        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        hp1 = Biquad::HighPass(fs, loHz, Q1);
        hp2 = Biquad::HighPass(fs, loHz, Q2);
        lp1 = Biquad::LowPass(fs, hiHz, Q1);
        lp2 = Biquad::LowPass(fs, hiHz, Q2);
        hp1.reset(); hp2.reset(); lp1.reset(); lp2.reset();
    }

    void updateBand(float lo, float hi) {
        loHz = lo; hiHz = hi;
        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        UpdateCoeffsPreserveState(hp1, Biquad::HighPass(fs, loHz, Q1));
        UpdateCoeffsPreserveState(hp2, Biquad::HighPass(fs, loHz, Q2));
        UpdateCoeffsPreserveState(lp1, Biquad::LowPass(fs, hiHz, Q1));
        UpdateCoeffsPreserveState(lp2, Biquad::LowPass(fs, hiHz, Q2));
    }

    float process(float x) {
        if (!enabled || amount <= 0.0f) return x;

        float band = hp2.process(hp1.process(x));
        band = lp2.process(lp1.process(band));

        float d = std::max(1.0f, drive);
        float distorted = std::tanh(band * d) / std::tanh(d);

        float harmonicsOnly = distorted - band;
        return x + amount * harmonicsOnly;
    }
};

// Sliding-window max for lookahead limiter
struct MaxQueue {
    struct Node { int idx; float v; };
    std::deque<Node> dq;
    int idx = 0;

    void clear() { dq.clear(); idx = 0; }

    void push(float v) {
        while (!dq.empty() && dq.back().v <= v) dq.pop_back();
        dq.push_back({ idx, v });
        idx++;
    }

    void popOld(int oldestIdx) {
        while (!dq.empty() && dq.front().idx < oldestIdx) dq.pop_front();
    }

    float max() const { return dq.empty() ? 0.0f : dq.front().v; }
};

// True-peak-ish oversampled lookahead limiter
struct TruePeakLimiter {
    float fs = 48000.0f;
    bool enabled = true;

    // "True-peak-ish" ceiling. Keep this conservative; map to safe deviation with TX gain.
    float ceilingDb = -3.0f;
    float ceiling = 0.7079458f;

    float lookaheadMs = 6.0f;
    float attackMs = 0.5f;
    float releaseMs = 60.0f;

    int win = 0;
    int sampleIndex = 0;

    float aA = 0.0f, aR = 0.0f;
    float g = 1.0f;

    std::deque<float> buf;
    MaxQueue mq;

    // Oversampled peak estimation (fractional-delay windowed-sinc interpolation).
    // This is heavier than simple peak, but helps keep inter-sample overs from clipping.
    int os = 8;          // 1,2,4,8 (default 8 for "big but safe")
    int taps = 48;       // per-phase taps
    float fc = 0.45f;    // cutoff (normalized to Nyquist)
    std::vector<float> table; // [os][taps]
    std::vector<float> hist;  // ring buffer of last taps samples
    int histPos = 0;

    float lastLookaheadMs = -1.0f;
    float lastAttackMs = -1.0f;
    float lastReleaseMs = -1.0f;
    int   lastOs = -1;

    void buildTable() {
        int O = Clamp(os, 1, 8);
        os = O;

        // Choose taps (heavier for higher OS)
        taps = (os >= 8) ? 64 : (os >= 4 ? 48 : (os == 2 ? 40 : 1));
        if (os == 1) taps = 1;

        table.assign((size_t)os * (size_t)taps, 0.0f);
        hist.assign((size_t)taps, 0.0f);
        histPos = 0;

        if (os == 1) {
            table[0] = 1.0f;
            return;
        }

        int mid = taps / 2;

        auto sinc = [](float x) -> float {
            if (std::fabs(x) < 1e-8f) return 1.0f;
            return std::sin(kPi * x) / (kPi * x);
        };

        // Blackman window
        auto winBlackman = [&](int n) -> float {
            return 0.42f
                - 0.5f * std::cos(2.0f * kPi * (float)n / (float)(taps - 1))
                + 0.08f * std::cos(4.0f * kPi * (float)n / (float)(taps - 1));
        };

        for (int p = 0; p < os; p++) {
            float d = (float)p / (float)os; // fractional delay in [0,1)
            float sum = 0.0f;

            for (int n = 0; n < taps; n++) {
                float k = (float)(n - mid) - d;
                // windowed-sinc lowpass (2*fc factor)
                float h = (2.0f * fc) * sinc(2.0f * fc * k);
                h *= winBlackman(n);
                table[(size_t)p * (size_t)taps + (size_t)n] = h;
                sum += h;
            }

            // Normalize each phase to unity DC gain
            if (std::fabs(sum) > 1e-12f) {
                for (int n = 0; n < taps; n++) {
                    table[(size_t)p * (size_t)taps + (size_t)n] /= sum;
                }
            }
        }
    }

    void resetState(bool keepGain = false) {
        buf.clear();
        mq.clear();
        sampleIndex = 0;
        if (!keepGain) g = 1.0f;
        std::fill(hist.begin(), hist.end(), 0.0f);
        histPos = 0;
    }

    void init(float Fs) {
        fs = Fs;
        setCeilingDb(ceilingDb);

        win = std::max(8, (int)std::round(fs * (lookaheadMs / 1000.0f)));

        aA = std::exp(-1.0f / (fs * (attackMs / 1000.0f)));
        aR = std::exp(-1.0f / (fs * (releaseMs / 1000.0f)));

        buildTable();
        resetState(false);

        lastLookaheadMs = lookaheadMs;
        lastAttackMs = attackMs;
        lastReleaseMs = releaseMs;
        lastOs = os;
    }

    void setCeilingDb(float db) { ceilingDb = db; ceiling = dbToLin(db); }

    void ensureConfig() {
        // If user changed key parameters live, re-init cleanly (safer than half-updating).
        if (lastOs != os ||
            std::fabs(lastLookaheadMs - lookaheadMs) > 0.05f ||
            std::fabs(lastAttackMs - attackMs) > 0.05f ||
            std::fabs(lastReleaseMs - releaseMs) > 0.05f)
        {
            init(fs);
        }
    }

    float convolvePhase(int phase) const {
        // phase in [0, os-1] representing delay = phase/os samples (fractional)
        float y = 0.0f;
        const size_t base = (size_t)phase * (size_t)taps;
        for (int n = 0; n < taps; n++) {
            int idx = histPos - n;
            while (idx < 0) idx += taps;
            y += table[base + (size_t)n] * hist[(size_t)idx];
        }
        return y;
    }

    float peakEstimate(float xNew) {
        // push into history
        histPos++;
        if (histPos >= taps) histPos = 0;
        hist[(size_t)histPos] = xNew;

        float maxv = std::fabs(xNew);
        if (os <= 1) return maxv;

        // Evaluate points between previous and current sample:
        // time = (i-1 + m/os) => delay d = 1 - m/os => phase = os - m
        for (int m = 1; m < os; m++) {
            int phase = os - m;
            float y = convolvePhase(phase);
            maxv = std::max(maxv, std::fabs(y));
        }
        return maxv;
    }

    float process(float x) {
        if (!enabled) return x;
        ensureConfig();

        buf.push_back(x);
        float pk = peakEstimate(x);
        mq.push(pk);
        sampleIndex++;

        if ((int)buf.size() < win) return 0.0f;

        int oldestIdx = sampleIndex - (int)buf.size();
        mq.popOld(oldestIdx);
        float peak = mq.max();

        float target = 1.0f;
        if (peak > ceiling && peak > 1e-9f) target = ceiling / peak;

        // Smooth gain: fast attack when reducing gain, slower release when restoring.
        if (target < g) g = aA * g + (1.0f - aA) * target;
        else            g = aR * g + (1.0f - aR) * target;

        float y = buf.front();
        buf.pop_front();
        return y * g;
    }
};

// Plosive controller: detect LF bursts and temporarily raise HPF
struct PlosiveController {
    float fs = 48000.0f;
    bool enabled = true;

    float baseHpHz = 120.0f;
    float maxHpHz = 240.0f;
    float sense = 0.9f;
    float releaseMs = 140.0f;

    Biquad lp;
    EnvFollower env;
    float hpCurrent = 120.0f;

    void init(float Fs) {
        fs = Fs;
        lp = Biquad::LowPass(fs, 120.0f, 0.7071f);
        lp.reset();
        env.set(fs, 2.0f, 120.0f);
        env.env = 0;
        hpCurrent = baseHpHz;
    }

    float updateHp(float x) {
        if (!enabled) return baseHpHz;

        float lf = lp.process(x);
        float e = env.process(std::fabs(lf));
        float eDb = linToDb(e);

        float over = eDb - (-25.0f + (1.0f - sense) * 8.0f);
        float targetHp = baseHpHz;

        if (over > 0.0f) {
            float t = Clamp(over / 18.0f, 0.0f, 1.0f);
            targetHp = baseHpHz + t * (maxHpHz - baseHpHz);
        }

        float aR = std::exp(-1.0f / (fs * (releaseMs / 1000.0f)));
        if (targetHp > hpCurrent) hpCurrent = 0.6f * hpCurrent + 0.4f * targetHp;
        else hpCurrent = aR * hpCurrent + (1.0f - aR) * targetHp;

        hpCurrent = Clamp(hpCurrent, baseHpHz, maxHpHz);
        return hpCurrent;
    }
};

// Linkwitz-Riley 4th order crossover split
struct LR4Split {
    float fs = 48000.0f;
    float fc = 1000.0f;
    Biquad lp1, lp2, hp1, hp2;

    void init(float Fs, float Fc) {
        fs = Fs; fc = Fc;
        lp1 = Biquad::LowPass(fs, fc, 0.7071f);
        lp2 = Biquad::LowPass(fs, fc, 0.7071f);
        hp1 = Biquad::HighPass(fs, fc, 0.7071f);
        hp2 = Biquad::HighPass(fs, fc, 0.7071f);
        lp1.reset(); lp2.reset(); hp1.reset(); hp2.reset();
    }

    void update(float Fc) {
        fc = Fc;
        UpdateCoeffsPreserveState(lp1, Biquad::LowPass(fs, fc, 0.7071f));
        UpdateCoeffsPreserveState(lp2, Biquad::LowPass(fs, fc, 0.7071f));
        UpdateCoeffsPreserveState(hp1, Biquad::HighPass(fs, fc, 0.7071f));
        UpdateCoeffsPreserveState(hp2, Biquad::HighPass(fs, fc, 0.7071f));
    }

    inline void process(float x, float& low, float& high) {
        low = lp2.process(lp1.process(x));
        high = hp2.process(hp1.process(x));
    }
};

// 4-band multiband compressor
struct Multiband4 {
    float fs = 48000.0f;
    bool enabled = true;

    float f1 = 160.0f;
    float f2 = 650.0f;
    float f3 = 2100.0f;

    float density = 0.85f;
    float makeupDb = 2.0f;

    Compressor cL, cLM, cHM, cH;
    LR4Split s1, s2, s3;

    void init(float Fs) {
        fs = Fs;
        s1.init(fs, f1);
        s2.init(fs, f2);
        s3.init(fs, f3);

        cL.envAttackMs = 10;  cL.envReleaseMs = 220; cL.gAttackMs = 6; cL.gReleaseMs = 140;
        cLM.envAttackMs = 6;  cLM.envReleaseMs = 180; cLM.gAttackMs = 4; cLM.gReleaseMs = 120;
        cHM.envAttackMs = 3;  cHM.envReleaseMs = 140; cHM.gAttackMs = 3; cHM.gReleaseMs = 90;
        cH.envAttackMs = 2;   cH.envReleaseMs = 120; cH.gAttackMs = 2; cH.gReleaseMs = 80;

        cL.init(fs); cLM.init(fs); cHM.init(fs); cH.init(fs);
        updateFromDensity();
    }

    void updateXovers(float F1, float F2, float F3) {
        f1 = Clamp(F1, 80.0f, 400.0f);
        f2 = Clamp(F2, f1 + 80.0f, 1400.0f);
        f3 = Clamp(F3, f2 + 200.0f, 3800.0f);
        s1.update(f1); s2.update(f2); s3.update(f3);
    }

    void updateFromDensity() {
        float d = Clamp(density, 0.0f, 1.0f);

        float thrBase = -34.0f + d * 10.0f;
        float ratBase = 1.8f + d * 4.2f;

        cL.enabled = true;
        cL.threshDb = thrBase - 2.0f;
        cL.ratio = ratBase * 0.85f;
        cL.setMakeupDb(0.0f);

        cLM.enabled = true;
        cLM.threshDb = thrBase;
        cLM.ratio = ratBase;
        cLM.setMakeupDb(0.0f);

        cHM.enabled = true;
        cHM.threshDb = thrBase + 1.0f;
        cHM.ratio = ratBase * 1.05f;
        cHM.setMakeupDb(0.0f);

        cH.enabled = true;
        cH.threshDb = thrBase + 2.0f;
        cH.ratio = 1.6f + d * 2.0f;
        cH.setMakeupDb(0.0f);
    }

    float process(float x) {
        if (!enabled) return x;

        float low = 0, rest = 0;
        float lm = 0;
        float hm = 0, high = 0;

        s1.process(x, low, rest);
        s2.process(rest, lm, rest);
        s3.process(rest, hm, high);

        low = cL.process(low);
        lm = cLM.process(lm);
        hm = cHM.process(hm);
        high = cH.process(high);

        float y = (low + lm + hm + high) * dbToLin(makeupDb);
        return y;
    }
};

static inline float satTanh(float x, float drive) {
    drive = std::max(1.0f, drive);
    float t = std::tanh(x * drive);
    float n = std::tanh(drive);
    return (n > 1e-9f) ? (t / n) : t;
}

// Phase rotator (magnitude-flat allpass cascade).
// Broadcast trick: rotates phase to reduce crest factor (peaks) without changing EQ,
// allowing "bigger" perceived loudness before limiting/clipping.
struct PhaseRotator {
    struct AP1 {
        float a = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;

        void reset() { x1 = y1 = 0.0f; }

        void set(float fs, float fHz) {
            fHz = Clamp(fHz, 20.0f, fs * 0.45f);
            float t = std::tan(kPi * (fHz / fs));
            // stable allpass coefficient
            float aa = (t - 1.0f) / (t + 1.0f);
            a = Clamp(aa, -0.9999f, 0.9999f);
        }

        float process(float x) {
            // 1st order allpass: y[n] = a*x[n] + x[n-1] - a*y[n-1]
            float y = a * x + x1 - a * y1;
            x1 = x;
            y1 = y;
            return y;
        }
    };

    float fs = 48000.0f;
    bool enabled = true;

    // section corner freqs (spread across voice band)
    float f1 = 120.0f, f2 = 320.0f, f3 = 900.0f, f4 = 2100.0f;

    AP1 s1, s2, s3, s4;

    void init(float Fs) {
        fs = Fs;
        s1.set(fs, f1);
        s2.set(fs, f2);
        s3.set(fs, f3);
        s4.set(fs, f4);
        reset();
    }

    void setFreqs(float F1, float F2, float F3, float F4) {
        f1 = F1; f2 = F2; f3 = F3; f4 = F4;
        s1.set(fs, f1);
        s2.set(fs, f2);
        s3.set(fs, f3);
        s4.set(fs, f4);
    }

    void reset() { s1.reset(); s2.reset(); s3.reset(); s4.reset(); }

    float process(float x) {
        if (!enabled) return x;
        float y = s1.process(x);
        y = s2.process(y);
        y = s3.process(y);
        y = s4.process(y);
        return y;
    }
};

// Monitor sim
struct MonitorSim {
    float fs = 48000.0f;
    bool enabled = false;
    Biquad hp, lp1, lp2, honk;

    void init(float Fs) {
        fs = Fs;
        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        hp = Biquad::HighPass(fs, 280.0f, 0.7071f);
        lp1 = Biquad::LowPass(fs, 3200.0f, Q1);
        lp2 = Biquad::LowPass(fs, 3200.0f, Q2);
        honk = Biquad::Peaking(fs, 1200.0f, 1.2f, +2.5f);
        hp.reset(); lp1.reset(); lp2.reset(); honk.reset();
    }

    float process(float x) {
        if (!enabled) return x;
        float y = hp.process(x);
        y = lp2.process(lp1.process(y));
        y = honk.process(y);
        return y;
    }
};

// High-quality resampler (bandlimited windowed-sinc, polyphase table)
struct SincResampler {
    double inRate = 48000.0, outRate = 48000.0;
    double step = 1.0;
    double pos = 0.0;

    int phases = 1024;
    int taps = 48;
    std::vector<float> table;

    void buildTable() {
        table.assign((size_t)phases * (size_t)taps, 0.0f);

        double r = outRate / inRate;
        double cutoff = 0.475 * std::min(1.0, r);

        int mid = taps / 2;
        for (int p = 0; p < phases; p++) {
            double frac = (double)p / (double)phases;
            double sum = 0.0;
            for (int n = 0; n < taps; n++) {
                int k = n - mid;
                double x = (double)k - frac;

                double sinc = 1.0;
                double arg = 2.0 * kPi * cutoff * x;
                if (std::fabs(arg) > 1e-12) sinc = std::sin(arg) / arg;

                double w = 0.42 - 0.5 * std::cos(2.0 * kPi * (double)n / (double)(taps - 1))
                    + 0.08 * std::cos(4.0 * kPi * (double)n / (double)(taps - 1));

                double h = 2.0 * cutoff * sinc * w;
                table[(size_t)p * (size_t)taps + (size_t)n] = (float)h;
                sum += h;
            }
            if (std::fabs(sum) > 1e-12) {
                for (int n = 0; n < taps; n++) {
                    table[(size_t)p * (size_t)taps + (size_t)n] = (float)(table[(size_t)p * (size_t)taps + (size_t)n] / (float)sum);
                }
            }
        }
    }

    void setRates(double inR, double outR) {
        inRate = inR; outRate = outR;
        step = inRate / outRate;
        pos = 0.0;
        buildTable();
    }

    void produce(std::deque<float>& inBuf, std::vector<float>& out) {
        out.clear();
        if (inBuf.size() < (size_t)taps + 2) return;

        while ((pos + (double)taps) < (double)inBuf.size()) {
            int iPos = (int)pos;
            double frac = pos - (double)iPos;
            int p = (int)(frac * (double)phases);
            p = Clamp(p, 0, phases - 1);

            const float* h = &table[(size_t)p * (size_t)taps];
            float acc = 0.0f;
            for (int n = 0; n < taps; n++) acc += h[n] * inBuf[(size_t)iPos + (size_t)n];
            out.push_back(acc);
            pos += step;
        }

        int consume = (int)pos;
        if (consume > 0) {
            consume = std::min<int>(consume, (int)inBuf.size());
            for (int k = 0; k < consume; k++) inBuf.pop_front();
            pos -= (double)consume;
        }
    }
};

// --------------------------- Main processor ---------------------------

enum class Mode { FM = 1, AM = 2, SSB = 3 };

struct Preset {
    float hpHz = 120.0f;
    float lpHz = 3200.0f;

    bool eqOn = true;
    float lowShelfDb = +2.0f;
    float eqNasalDb = -3.0f;
    float eqPresDb = +4.0f;
    float highShelfDb = +1.5f;

    bool phaseRotOn = true;

    bool plosiveOn = true;
    float plosiveSense = 0.9f;
    float plosiveMaxHpHz = 240.0f;

    bool gateOn = true;
    float gateOpenDb = -55.0f;
    float gateCloseDb = -60.0f;
    float gateRatio = 4.0f;
    float gateFloorDb = -35.0f;
    float gateHoldMs = 60.0f;

    bool agcOn = true;
    float agcTargetDb = -18.0f;
    float agcMaxGainDb = 18.0f;

    bool mbOn = true;
    float mbDensity = 0.85f;
    float mbMakeupDb = 2.0f;
    float mbF1 = 160.0f, mbF2 = 650.0f, mbF3 = 2100.0f;

    bool deessOn = true;
    float deessThreshDb = -30.0f;
    float deessAmt = 0.6f;

    bool excOn = true;
    float excAmt = 0.25f;
    float excDrive = 2.2f;
    float excLoHz = 1200.0f;
    float excHiHz = 3400.0f;

    float satDrive = 1.5f;

    bool limOn = true;
    float limCeilDb = -3.0f;
    float limLookaheadMs = 6.0f;
    float limReleaseMs = 60.0f;
    int limOversample = 4;

    float outGainDb = 10.0f;

    bool monSim = false;
};

struct ParamDesc {
    const char* name;
    float* value;
    float step;
    float bigStep;
    float minV;
    float maxV;
};

struct Processor {
    float fs = 48000.0f;
    Mode mode = Mode::AM;

    Preset fm, am, ssb;
    Preset* cur = nullptr;

    Biquad hpf, lpf1, lpf2;

    Biquad lowShelf, nasal, pres, highShelf;

    PhaseRotator phrot;

    PlosiveController plosive;
    SmartGate gate;
    LevelerAGC agc;
    Multiband4 mb;
    Exciter exc;
    DeEsser deess;
    TruePeakLimiter lim;
    MonitorSim mon;

    Biquad postLP1, postLP2;

    std::vector<ParamDesc> params;
    int paramIndex = 0;

    void init(float Fs, Mode m) {
        fs = Fs;

        // "Godlike" defaults: dense + clear without illegal over-mod.
// You still MUST set your radio's mic gain/deviation so this stays within spec.
        fm = Preset{};
        fm.hpHz = 140; fm.lpHz = 3000;
        fm.eqOn = true;
        fm.lowShelfDb = +3.0f; fm.eqNasalDb = -3.0f; fm.eqPresDb = +4.0f; fm.highShelfDb = +1.8f;
        fm.phaseRotOn = true;
        fm.plosiveOn = true; fm.plosiveSense = 0.92f; fm.plosiveMaxHpHz = 280.0f;
        fm.gateOn = true; fm.gateOpenDb = -55.0f; fm.gateCloseDb = -60.0f; fm.gateRatio = 4.0f; fm.gateFloorDb = -35.0f; fm.gateHoldMs = 70.0f;
        fm.agcOn = true; fm.agcTargetDb = -18.0f; fm.agcMaxGainDb = 18.0f;
        fm.mbOn = true; fm.mbDensity = 0.92f; fm.mbMakeupDb = 3.0f;
        fm.excOn = true; fm.excAmt = 0.22f; fm.excDrive = 2.0f; fm.excLoHz = 1200.0f; fm.excHiHz = 2800.0f;
        fm.deessOn = true; fm.deessThreshDb = -30.0f; fm.deessAmt = 0.6f;
        fm.satDrive = 1.4f;
        fm.limOn = true; fm.limCeilDb = -3.0f; fm.limLookaheadMs = 6.0f; fm.limReleaseMs = 80.0f; fm.limOversample = 8;
        fm.outGainDb = 12.0f;
        fm.monSim = false;

        am = Preset{};
        am.hpHz = 95; am.lpHz = 4200;
        am.eqOn = true;
        am.lowShelfDb = +3.0f; am.eqNasalDb = -3.0f; am.eqPresDb = +10.0f; am.highShelfDb = +1.2f;
        am.phaseRotOn = true;
        am.plosiveOn = true; am.plosiveSense = 0.90f; am.plosiveMaxHpHz = 260.0f;
        am.gateOn = true; am.gateOpenDb = -65.0f; am.gateCloseDb = -70.0f; am.gateRatio = 6.0f; am.gateFloorDb = -18.0f; am.gateHoldMs = 80.0f;
        am.agcOn = true; am.agcTargetDb = -18.0f; am.agcMaxGainDb = 18.0f;
        am.mbOn = true; am.mbDensity = 0.95f; am.mbMakeupDb = 3.5f;
        am.excOn = true; am.excAmt = 0.35f; am.excDrive = 2.8f; am.excLoHz = 1500.0f; am.excHiHz = 3800.0f;
        am.deessOn = false;
        am.satDrive = 1.8f;
        am.limOn = true; am.limCeilDb = -2.0f; am.limLookaheadMs = 6.0f; am.limReleaseMs = 90.0f; am.limOversample = 8;
        am.outGainDb = 14.0f;
        am.monSim = false;

        ssb = Preset{};
        ssb.hpHz = 200; ssb.lpHz = 2900;
        ssb.eqOn = true;
        ssb.lowShelfDb = +2.5f; ssb.eqNasalDb = -3.0f; ssb.eqPresDb = +4.5f; ssb.highShelfDb = +2.0f;
        ssb.phaseRotOn = true;
        ssb.plosiveOn = true; ssb.plosiveSense = 0.92f; ssb.plosiveMaxHpHz = 280.0f;
        ssb.gateOn = true; ssb.gateOpenDb = -55.0f; ssb.gateCloseDb = -60.0f; ssb.gateRatio = 4.0f; ssb.gateFloorDb = -35.0f; ssb.gateHoldMs = 70.0f;
        ssb.agcOn = true; ssb.agcTargetDb = -18.0f; ssb.agcMaxGainDb = 18.0f;
        ssb.mbOn = true; ssb.mbDensity = 0.90f; ssb.mbMakeupDb = 3.0f;
        ssb.excOn = true; ssb.excAmt = 0.18f; ssb.excDrive = 2.0f; ssb.excLoHz = 1200.0f; ssb.excHiHz = 2600.0f;
        ssb.deessOn = true; ssb.deessThreshDb = -30.0f; ssb.deessAmt = 0.6f;
        ssb.satDrive = 1.35f;
        ssb.limOn = true; ssb.limCeilDb = -3.0f; ssb.limLookaheadMs = 6.0f; ssb.limReleaseMs = 80.0f; ssb.limOversample = 8;
        ssb.outGainDb = 12.0f;
        ssb.monSim = false;

        setMode(m);
    }

    void setMode(Mode m) {
        mode = m;
        if (m == Mode::FM) cur = &fm;
        else if (m == Mode::AM) cur = &am;
        else cur = &ssb;

        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        hpf = Biquad::HighPass(fs, cur->hpHz, 0.7071f);
        lpf1 = Biquad::LowPass(fs, cur->lpHz, Q1);
        lpf2 = Biquad::LowPass(fs, cur->lpHz, Q2);
        hpf.reset(); lpf1.reset(); lpf2.reset();

        lowShelf = Biquad::LowShelf(fs, 160.0f, 0.7071f, cur->lowShelfDb);
        nasal = Biquad::Peaking(fs, 800.0f, 1.2f, cur->eqNasalDb);
        pres = Biquad::Peaking(fs, 2400.0f, 1.0f, cur->eqPresDb);
        highShelf = Biquad::HighShelf(fs, 3600.0f, 0.7071f, cur->highShelfDb);
        lowShelf.reset(); nasal.reset(); pres.reset(); highShelf.reset();

        phrot.enabled = cur->phaseRotOn;
        if (mode == Mode::FM)      phrot.setFreqs(120.0f, 320.0f, 900.0f, 2100.0f);
        else if (mode == Mode::AM) phrot.setFreqs(100.0f, 260.0f, 800.0f, 2000.0f);
        else                      phrot.setFreqs(140.0f, 360.0f, 1000.0f, 2200.0f);
        phrot.init(fs);


        plosive.enabled = cur->plosiveOn;
        plosive.baseHpHz = cur->hpHz;
        plosive.maxHpHz = cur->plosiveMaxHpHz;
        plosive.sense = cur->plosiveSense;
        plosive.init(fs);

        gate.enabled = cur->gateOn;
        gate.openThreshDb = cur->gateOpenDb;
        gate.closeThreshDb = cur->gateCloseDb;
        gate.ratio = std::max(1.0f, cur->gateRatio);
        gate.floorDb = cur->gateFloorDb;
        gate.holdMs = cur->gateHoldMs;
        gate.init(fs);

        agc.enabled = cur->agcOn;
        agc.targetDb = cur->agcTargetDb;
        agc.maxGainDb = cur->agcMaxGainDb;
        agc.init(fs);

        mb.enabled = cur->mbOn;
        mb.density = cur->mbDensity;
        mb.makeupDb = cur->mbMakeupDb;
        mb.init(fs);
        mb.updateXovers(cur->mbF1, cur->mbF2, cur->mbF3);
        mb.updateFromDensity();

        exc.enabled = cur->excOn;
        exc.amount = cur->excAmt;
        exc.drive = cur->excDrive;
        exc.loHz = cur->excLoHz;
        exc.hiHz = cur->excHiHz;
        exc.init(fs);

        deess.enabled = cur->deessOn;
        deess.threshDb = cur->deessThreshDb;
        deess.detAmount = cur->deessAmt;
        deess.init(fs);

        lim.enabled = cur->limOn;
        lim.setCeilingDb(cur->limCeilDb);
        lim.lookaheadMs = cur->limLookaheadMs;
        lim.releaseMs = cur->limReleaseMs;
        lim.os = Clamp(cur->limOversample, 1, 8);
        lim.attackMs = 0.5f;
        lim.init(fs);

        mon.enabled = cur->monSim;
        mon.init(fs);

        float postFc = std::min(cur->lpHz * 0.98f, fs * 0.45f);
        postLP1 = Biquad::LowPass(fs, postFc, Q1);
        postLP2 = Biquad::LowPass(fs, postFc, Q2);
        postLP1.reset(); postLP2.reset();

        rebuildParamList();
    }

    void rebuildParamList() {
        params.clear();
        params.push_back({ "outGain(dB)", &cur->outGainDb, 0.5f, 2.0f, -30.0f, +30.0f });

        params.push_back({ "HP base(Hz)", &cur->hpHz, 5.0f, 25.0f, 40.0f, 350.0f });
        params.push_back({ "LP(Hz)", &cur->lpHz, 25.0f, 150.0f, 2000.0f, 6000.0f });

        params.push_back({ "LowShelf(dB)", &cur->lowShelfDb, 0.5f, 2.0f, -12.0f, +12.0f });
        params.push_back({ "Nasal(dB)", &cur->eqNasalDb, 0.5f, 2.0f, -12.0f, +12.0f });
        params.push_back({ "Presence(dB)", &cur->eqPresDb, 0.5f, 2.0f, -12.0f, +18.0f });
        params.push_back({ "HighShelf(dB)", &cur->highShelfDb, 0.5f, 2.0f, -12.0f, +12.0f });

        params.push_back({ "Plosive sense", &cur->plosiveSense, 0.05f, 0.15f, 0.0f, 1.0f });
        params.push_back({ "Plosive maxHP", &cur->plosiveMaxHpHz, 5.0f, 25.0f, 160.0f, 320.0f });

        params.push_back({ "Gate open(dB)", &cur->gateOpenDb, 1.0f, 4.0f, -90.0f, -10.0f });
        params.push_back({ "Gate close(dB)", &cur->gateCloseDb, 1.0f, 4.0f, -95.0f, -15.0f });
        params.push_back({ "Gate ratio", &cur->gateRatio, 0.5f, 2.0f, 1.0f, 12.0f });
        params.push_back({ "Gate floor(dB)", &cur->gateFloorDb, 1.0f, 4.0f, -60.0f, 0.0f });
        params.push_back({ "Gate hold(ms)", &cur->gateHoldMs, 5.0f, 25.0f, 0.0f, 250.0f });

        params.push_back({ "AGC target(dB)", &cur->agcTargetDb, 0.5f, 2.0f, -30.0f, -6.0f });
        params.push_back({ "AGC maxG(dB)", &cur->agcMaxGainDb, 0.5f, 2.0f, 0.0f, 30.0f });

        params.push_back({ "MB density", &cur->mbDensity, 0.02f, 0.08f, 0.0f, 1.0f });
        params.push_back({ "MB makeup(dB)", &cur->mbMakeupDb, 0.5f, 2.0f, -12.0f, 12.0f });
        params.push_back({ "MB f1(Hz)", &cur->mbF1, 5.0f, 25.0f, 80.0f, 350.0f });
        params.push_back({ "MB f2(Hz)", &cur->mbF2, 10.0f, 50.0f, 250.0f, 1400.0f });
        params.push_back({ "MB f3(Hz)", &cur->mbF3, 25.0f, 100.0f, 800.0f, 3800.0f });

        params.push_back({ "DeEss thr(dB)", &cur->deessThreshDb, 1.0f, 4.0f, -60.0f, -10.0f });
        params.push_back({ "DeEss amt", &cur->deessAmt, 0.05f, 0.2f, 0.0f, 1.0f });

        params.push_back({ "Exc amt", &cur->excAmt, 0.02f, 0.1f, 0.0f, 1.0f });
        params.push_back({ "Exc drive", &cur->excDrive, 0.2f, 0.8f, 1.0f, 6.0f });

        params.push_back({ "Sat drive", &cur->satDrive, 0.05f, 0.2f, 1.0f, 4.0f });

        params.push_back({ "TP ceil(dB)", &cur->limCeilDb, 0.5f, 2.0f, -12.0f, 0.0f });
        params.push_back({ "TP look(ms)", &cur->limLookaheadMs, 1.0f, 3.0f, 2.0f, 30.0f });
        params.push_back({ "TP rel(ms)", &cur->limReleaseMs, 5.0f, 20.0f, 20.0f, 300.0f });
        // oversample is int, not in param list for safety

        paramIndex = std::min(paramIndex, (int)params.size() - 1);
    }

    void applyPresetToDSP() {
        setMode(mode);
    }

    float processOne(float x) {
        // Plosive dynamic HPF
        plosive.enabled = cur->plosiveOn;
        plosive.baseHpHz = cur->hpHz;
        plosive.maxHpHz = cur->plosiveMaxHpHz;
        plosive.sense = cur->plosiveSense;

        float dynHp = plosive.updateHp(x);
        UpdateCoeffsPreserveState(hpf, Biquad::HighPass(fs, dynHp, 0.7071f));

        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        UpdateCoeffsPreserveState(lpf1, Biquad::LowPass(fs, cur->lpHz, Q1));
        UpdateCoeffsPreserveState(lpf2, Biquad::LowPass(fs, cur->lpHz, Q2));

        float y = hpf.process(x);
        y = lpf2.process(lpf1.process(y));

        // AGC (slow leveler)
        agc.enabled = cur->agcOn;
        agc.targetDb = cur->agcTargetDb;
        agc.maxGainDb = cur->agcMaxGainDb;
        y = agc.process(y);

        // Gate/expander with hysteresis + hold (after AGC keeps noise down without AGC "running away" on silence)
        gate.enabled = cur->gateOn;
        gate.openThreshDb = cur->gateOpenDb;
        gate.closeThreshDb = std::min(cur->gateCloseDb, cur->gateOpenDb - 1.0f);
        gate.ratio = std::max(1.0f, cur->gateRatio);
        gate.floorDb = cur->gateFloorDb;
        gate.holdMs = cur->gateHoldMs;
        y = gate.process(y);


        // EQ
        if (cur->eqOn) {
            UpdateCoeffsPreserveState(lowShelf, Biquad::LowShelf(fs, 160.0f, 0.7071f, cur->lowShelfDb));
            UpdateCoeffsPreserveState(nasal, Biquad::Peaking(fs, 800.0f, 1.2f, cur->eqNasalDb));
            UpdateCoeffsPreserveState(pres, Biquad::Peaking(fs, 2400.0f, 1.0f, cur->eqPresDb));
            UpdateCoeffsPreserveState(highShelf, Biquad::HighShelf(fs, 3600.0f, 0.7071f, cur->highShelfDb));
            y = highShelf.process(pres.process(nasal.process(lowShelf.process(y))));
        }

        // Phase rotation (reduces crest factor before density stages)
        phrot.enabled = cur->phaseRotOn;
        y = phrot.process(y);


        // Multiband
        mb.enabled = cur->mbOn;
        mb.density = cur->mbDensity;
        mb.makeupDb = cur->mbMakeupDb;
        mb.updateXovers(cur->mbF1, cur->mbF2, cur->mbF3);
        mb.updateFromDensity();
        y = mb.process(y);

        // Exciter
        exc.enabled = cur->excOn;
        exc.amount = cur->excAmt;
        exc.drive = cur->excDrive;
        exc.updateBand(cur->excLoHz, cur->excHiHz);
        y = exc.process(y);

        // De-esser
        deess.enabled = cur->deessOn;
        deess.threshDb = cur->deessThreshDb;
        deess.detAmount = cur->deessAmt;
        if (mode == Mode::AM) deess.updateBand(3800.0f, 7500.0f);
        else                 deess.updateBand(4500.0f, 8000.0f);
        y = deess.process(y);

        // Saturation
        y = satTanh(Clamp(y, -1.2f, 1.2f), cur->satDrive);

        // Cleanup LP
        float postFc = std::min(cur->lpHz * 0.98f, fs * 0.45f);
        UpdateCoeffsPreserveState(postLP1, Biquad::LowPass(fs, postFc, Q1));
        UpdateCoeffsPreserveState(postLP2, Biquad::LowPass(fs, postFc, Q2));
        y = postLP2.process(postLP1.process(y));

        // Output gain then limiter
        y *= dbToLin(cur->outGainDb);

        lim.enabled = cur->limOn;
        lim.setCeilingDb(cur->limCeilDb);
        lim.lookaheadMs = cur->limLookaheadMs;
        lim.releaseMs = cur->limReleaseMs;
        lim.os = Clamp(cur->limOversample, 1, 8);
        y = lim.process(y);

        mon.enabled = cur->monSim;
        y = mon.process(y);

        return y;
    }

    void processBlock(const float* in, float* out, size_t n) {
        // simple, stable block processing
        for (size_t i = 0; i < n; i++) out[i] = processOne(in[i]);
    }
};

// -------------------- INI save/load --------------------

static void SaveIni(const char* path, const Preset& fm, const Preset& am, const Preset& ssb) {
    auto writePreset = [&](std::ofstream& o, const char* name, const Preset& p) {
        o << "[" << name << "]\n";
        o << "hpHz=" << p.hpHz << "\n";
        o << "lpHz=" << p.lpHz << "\n";

        o << "eqOn=" << (p.eqOn ? 1 : 0) << "\n";
        o << "lowShelfDb=" << p.lowShelfDb << "\n";
        o << "eqNasalDb=" << p.eqNasalDb << "\n";
        o << "eqPresDb=" << p.eqPresDb << "\n";
        o << "highShelfDb=" << p.highShelfDb << "\n";
        o << "phaseRotOn=" << (p.phaseRotOn ? 1 : 0) << "\n";


        o << "plosiveOn=" << (p.plosiveOn ? 1 : 0) << "\n";
        o << "plosiveSense=" << p.plosiveSense << "\n";
        o << "plosiveMaxHpHz=" << p.plosiveMaxHpHz << "\n";

        o << "gateOn=" << (p.gateOn ? 1 : 0) << "\n";
        o << "gateOpenDb=" << p.gateOpenDb << "\n";
        o << "gateCloseDb=" << p.gateCloseDb << "\n";
        o << "gateRatio=" << p.gateRatio << "\n";
        o << "gateFloorDb=" << p.gateFloorDb << "\n";
        o << "gateHoldMs=" << p.gateHoldMs << "\n";

        o << "agcOn=" << (p.agcOn ? 1 : 0) << "\n";
        o << "agcTargetDb=" << p.agcTargetDb << "\n";
        o << "agcMaxGainDb=" << p.agcMaxGainDb << "\n";

        o << "mbOn=" << (p.mbOn ? 1 : 0) << "\n";
        o << "mbDensity=" << p.mbDensity << "\n";
        o << "mbMakeupDb=" << p.mbMakeupDb << "\n";
        o << "mbF1=" << p.mbF1 << "\n";
        o << "mbF2=" << p.mbF2 << "\n";
        o << "mbF3=" << p.mbF3 << "\n";

        o << "deessOn=" << (p.deessOn ? 1 : 0) << "\n";
        o << "deessThreshDb=" << p.deessThreshDb << "\n";
        o << "deessAmt=" << p.deessAmt << "\n";

        o << "excOn=" << (p.excOn ? 1 : 0) << "\n";
        o << "excAmt=" << p.excAmt << "\n";
        o << "excDrive=" << p.excDrive << "\n";
        o << "excLoHz=" << p.excLoHz << "\n";
        o << "excHiHz=" << p.excHiHz << "\n";

        o << "satDrive=" << p.satDrive << "\n";

        o << "limOn=" << (p.limOn ? 1 : 0) << "\n";
        o << "limCeilDb=" << p.limCeilDb << "\n";
        o << "limLookaheadMs=" << p.limLookaheadMs << "\n";
        o << "limReleaseMs=" << p.limReleaseMs << "\n";
        o << "limOversample=" << p.limOversample << "\n";

        o << "outGainDb=" << p.outGainDb << "\n";
        o << "monSim=" << (p.monSim ? 1 : 0) << "\n\n";
    };

    std::ofstream o(path, std::ios::out | std::ios::trunc);
    if (!o) return;
    writePreset(o, "FM", fm);
    writePreset(o, "AM", am);
    writePreset(o, "SSB", ssb);
}

static bool ParseBool(const std::string& s) {
    return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
}

static bool LoadIni(const char* path, Preset& fm, Preset& am, Preset& ssb) {
    std::ifstream in(path);
    if (!in) return false;

    std::string line, section;
    Preset* cur = nullptr;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == ';' || line[0] == '#') continue;

        if (line.size() >= 3 && line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            if (section == "FM") cur = &fm;
            else if (section == "AM") cur = &am;
            else if (section == "SSB") cur = &ssb;
            else cur = nullptr;
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos || !cur) continue;

        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);

        auto f = [&]() { return (float)std::atof(v.c_str()); };
        auto i = [&]() { return (int)std::atoi(v.c_str()); };

        if (k == "hpHz") cur->hpHz = f();
        else if (k == "lpHz") cur->lpHz = f();

        else if (k == "eqOn") cur->eqOn = ParseBool(v);
        else if (k == "lowShelfDb") cur->lowShelfDb = f();
        else if (k == "eqNasalDb") cur->eqNasalDb = f();
        else if (k == "eqPresDb") cur->eqPresDb = f();
        else if (k == "highShelfDb") cur->highShelfDb = f();
        else if (k == "phaseRotOn") cur->phaseRotOn = ParseBool(v);

        else if (k == "plosiveOn") cur->plosiveOn = ParseBool(v);
        else if (k == "plosiveSense") cur->plosiveSense = f();
        else if (k == "plosiveMaxHpHz") cur->plosiveMaxHpHz = f();

        else if (k == "gateOn") cur->gateOn = ParseBool(v);
        else if (k == "gateOpenDb") cur->gateOpenDb = f();
        else if (k == "gateCloseDb") cur->gateCloseDb = f();
        else if (k == "gateRatio") cur->gateRatio = f();
        else if (k == "gateFloorDb") cur->gateFloorDb = f();
        else if (k == "gateHoldMs") cur->gateHoldMs = f();

        else if (k == "agcOn") cur->agcOn = ParseBool(v);
        else if (k == "agcTargetDb") cur->agcTargetDb = f();
        else if (k == "agcMaxGainDb") cur->agcMaxGainDb = f();

        else if (k == "mbOn") cur->mbOn = ParseBool(v);
        else if (k == "mbDensity") cur->mbDensity = f();
        else if (k == "mbMakeupDb") cur->mbMakeupDb = f();
        else if (k == "mbF1") cur->mbF1 = f();
        else if (k == "mbF2") cur->mbF2 = f();
        else if (k == "mbF3") cur->mbF3 = f();

        else if (k == "deessOn") cur->deessOn = ParseBool(v);
        else if (k == "deessThreshDb") cur->deessThreshDb = f();
        else if (k == "deessAmt") cur->deessAmt = f();

        else if (k == "excOn") cur->excOn = ParseBool(v);
        else if (k == "excAmt") cur->excAmt = f();
        else if (k == "excDrive") cur->excDrive = f();
        else if (k == "excLoHz") cur->excLoHz = f();
        else if (k == "excHiHz") cur->excHiHz = f();

        else if (k == "satDrive") cur->satDrive = f();

        else if (k == "limOn") cur->limOn = ParseBool(v);
        else if (k == "limCeilDb") cur->limCeilDb = f();
        else if (k == "limLookaheadMs") cur->limLookaheadMs = f();
        else if (k == "limReleaseMs") cur->limReleaseMs = f();
        else if (k == "limOversample") cur->limOversample = i();

        else if (k == "outGainDb") cur->outGainDb = f();
        else if (k == "monSim") cur->monSim = ParseBool(v);
    }
    return true;
}

// -------------------- MMCSS helper --------------------

static HANDLE SetMmcss(const wchar_t* taskName) {
    DWORD taskIndex = 0;
    return AvSetMmThreadCharacteristicsW(taskName, &taskIndex);
}

// -------------------- App entry --------------------

int wmain() {
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) Die("CoInitializeEx failed", hr);

    ComPtr<IMMDeviceEnumerator> en;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), (void**)&en);
    if (FAILED(hr)) Die("CoCreateInstance(MMDeviceEnumerator) failed", hr);

    ComPtr<IMMDevice> inDev = GetDefault(en.Get(), eCapture, eConsole);
    ComPtr<IMMDevice> outDev = GetDefault(en.Get(), eRender, eConsole);
    if (!inDev)  inDev = GetDefault(en.Get(), eCapture, eCommunications);
    if (!outDev) outDev = GetDefault(en.Get(), eRender, eCommunications);
    if (!inDev)  inDev = GetDefault(en.Get(), eCapture, eMultimedia);
    if (!outDev) outDev = GetDefault(en.Get(), eRender, eMultimedia);

    if (!inDev || !outDev) {
        std::fprintf(stderr, "Could not open default devices.\n");
        CoUninitialize();
        return 1;
    }

    std::wprintf(L"Input (default):  %s\n", GetDeviceName(inDev.Get()).c_str());
    std::wprintf(L"Output (default): %s\n", GetDeviceName(outDev.Get()).c_str());

    std::printf("\nHotkeys:\n");
    std::printf("  1=FM 2=AM 3=SSB | TAB select param | [ ] adjust | { } big adjust\n");
    std::printf("Toggles:\n");
    std::printf("  G gate | E EQ | P plosive | R phaseRot | A AGC | B multiband | D de-ess | X exciter | M monitor | S save | L load | Q quit\n");
    std::printf("Tools:\n");
    std::printf("  T = 1kHz tone (-12 dBFS, 5 sec) for TX gain calibration\n\n");

    ComPtr<IAudioClient> capClient, renClient;
    hr = inDev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&capClient);
    if (FAILED(hr)) Die("Activate capture IAudioClient failed", hr);
    hr = outDev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&renClient);
    if (FAILED(hr)) Die("Activate render IAudioClient failed", hr);

    WAVEFORMATEX* capFmt = nullptr;
    WAVEFORMATEX* renFmt = nullptr;
    hr = capClient->GetMixFormat(&capFmt);
    if (FAILED(hr)) Die("GetMixFormat(capture) failed", hr);
    hr = renClient->GetMixFormat(&renFmt);
    if (FAILED(hr)) Die("GetMixFormat(render) failed", hr);

    const bool capFloat = IsFloatFormat(capFmt);
    const bool capPcm16 = IsPcm16(capFmt);
    const bool renFloat = IsFloatFormat(renFmt);
    const bool renPcm16 = IsPcm16(renFmt);

    if (!(capFloat || capPcm16) || !(renFloat || renPcm16)) {
        std::fprintf(stderr, "Unsupported formats. Need float or 16-bit PCM.\n");
        CoTaskMemFree(capFmt); CoTaskMemFree(renFmt);
        CoUninitialize();
        return 1;
    }

    const double capRate = (double)capFmt->nSamplesPerSec;
    const double renRate = (double)renFmt->nSamplesPerSec;
    std::printf("Capture rate: %.0f Hz | Render rate: %.0f Hz\n", capRate, renRate);

    const REFERENCE_TIME hnsRequested = 10 * 10000; // 10ms
    hr = capClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
        hnsRequested, 0, capFmt, nullptr);
    if (FAILED(hr)) Die("capClient->Initialize failed", hr);

    hr = renClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
        hnsRequested, 0, renFmt, nullptr);
    if (FAILED(hr)) Die("renClient->Initialize failed", hr);

    UINT32 capBufFrames = 0, renBufFrames = 0;
    capClient->GetBufferSize(&capBufFrames);
    renClient->GetBufferSize(&renBufFrames);

    HANDLE hCapEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HANDLE hRenEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!hCapEvent || !hRenEvent) Die("CreateEvent failed", HRESULT_FROM_WIN32(GetLastError()));

    hr = capClient->SetEventHandle(hCapEvent);
    if (FAILED(hr)) Die("capClient->SetEventHandle failed", hr);
    hr = renClient->SetEventHandle(hRenEvent);
    if (FAILED(hr)) Die("renClient->SetEventHandle failed", hr);

    ComPtr<IAudioCaptureClient> cap;
    ComPtr<IAudioRenderClient>  ren;
    hr = capClient->GetService(__uuidof(IAudioCaptureClient), (void**)&cap);
    if (FAILED(hr)) Die("GetService(IAudioCaptureClient) failed", hr);
    hr = renClient->GetService(__uuidof(IAudioRenderClient), (void**)&ren);
    if (FAILED(hr)) Die("GetService(IAudioRenderClient) failed", hr);

    SpscRing capRing((size_t)(capRate * 0.5));
    SpscRing outRing((size_t)(renRate * 0.5));

    std::mutex procMtx;
    Processor proc;
    proc.init((float)renRate, Mode::AM);

    SincResampler rs;
    rs.setRates(capRate, renRate);
    std::deque<float> rsIn;
    std::vector<float> rsOut;
    rsOut.reserve(4096);

    std::atomic<int> toneSamplesLeft{ 0 };
    float tonePhase = 0.0f;

    std::atomic<float> meterDbfs{ -120.0f };

    hr = capClient->Start();
    if (FAILED(hr)) Die("capClient->Start failed", hr);
    hr = renClient->Start();
    if (FAILED(hr)) Die("renClient->Start failed", hr);

    std::thread capThread([&]() {
        HANDLE hTask = SetMmcss(L"Pro Audio");


        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        std::vector<float> tmp;
        tmp.reserve(capBufFrames);

        while (g_running.load()) {
            DWORD w = WaitForSingleObject(hCapEvent, 50);
            if (w != WAIT_OBJECT_0) continue;

            UINT32 packet = 0;
            HRESULT ehr = cap->GetNextPacketSize(&packet);
            if (FAILED(ehr)) continue;

            while (packet > 0) {
                BYTE* data = nullptr;
                UINT32 frames = 0;
                DWORD flags = 0;
                ehr = cap->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
                if (FAILED(ehr)) break;

                const int inCh = capFmt->nChannels;
                tmp.assign(frames, 0.0f);

                if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                    if (capFloat) {
                        const float* f = reinterpret_cast<const float*>(data);
                        for (UINT32 i = 0; i < frames; i++) {
                            float s = 0.0f;
                            for (int c = 0; c < inCh; c++) s += f[i * inCh + c];
                            tmp[i] = s / (float)inCh;
                        }
                    }
                    else {
                        const int16_t* p = reinterpret_cast<const int16_t*>(data);
                        for (UINT32 i = 0; i < frames; i++) {
                            float s = 0.0f;
                            for (int c = 0; c < inCh; c++) s += (float)p[i * inCh + c] / 32768.0f;
                            tmp[i] = s / (float)inCh;
                        }
                    }
                }

                cap->ReleaseBuffer(frames);
                capRing.push(tmp.data(), tmp.size());

                ehr = cap->GetNextPacketSize(&packet);
                if (FAILED(ehr)) break;
            }
        }

        if (hTask) AvRevertMmThreadCharacteristics(hTask);
        });

    std::thread dspThread([&]() {
        HANDLE hTask = SetMmcss(L"Pro Audio");


        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        std::vector<float> inTmp(2048);
        std::vector<float> outTmp;

        double rmsSum = 0.0;
        uint64_t rmsCount = 0;
        ULONGLONG lastMeterMs = GetTickCount64();

        while (g_running.load()) {
            size_t got = capRing.pop(inTmp.data(), inTmp.size());
            if (got == 0) { Sleep(1); continue; }

            for (size_t i = 0; i < got; i++) rsIn.push_back(inTmp[i]);

            rs.produce(rsIn, rsOut);
            if (rsOut.empty()) continue;

            outTmp.resize(rsOut.size());

            {
                std::lock_guard<std::mutex> lock(procMtx);

                int left = toneSamplesLeft.load();
                if (left > 0) {
                    float amp = dbToLin(-12.0f);
                    for (size_t i = 0; i < outTmp.size(); i++) {
                        if (left <= 0) { outTmp[i] = 0.0f; continue; }
                        float t = std::sinf(tonePhase) * amp;
                        tonePhase += 2.0f * kPi * (1000.0f / (float)renRate);
                        if (tonePhase > 2.0f * kPi) tonePhase -= 2.0f * kPi;

                        float y = t * dbToLin(proc.cur->outGainDb);
                        y = proc.lim.process(y);
                        outTmp[i] = y;
                        left--;
                    }
                    toneSamplesLeft.store(left);
                }
                else {
                    proc.processBlock(rsOut.data(), outTmp.data(), outTmp.size());
                }
            }

            outRing.push(outTmp.data(), outTmp.size());

            for (float y : outTmp) { rmsSum += (double)y * (double)y; rmsCount++; }
            ULONGLONG now = GetTickCount64();
            if (now - lastMeterMs >= 1000) {
                float rms = (rmsCount > 0) ? (float)std::sqrt(rmsSum / (double)rmsCount) : 0.0f;
                meterDbfs.store(linToDb(rms));
                rmsSum = 0.0; rmsCount = 0; lastMeterMs = now;
            }
        }

        if (hTask) AvRevertMmThreadCharacteristics(hTask);
        });

    std::thread renThread([&]() {
        HANDLE hTask = SetMmcss(L"Pro Audio");


        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        std::vector<float> tmp(4096);

        while (g_running.load()) {
            DWORD w = WaitForSingleObject(hRenEvent, 50);
            if (w != WAIT_OBJECT_0) continue;

            UINT32 padding = 0;
            HRESULT ehr = renClient->GetCurrentPadding(&padding);
            if (FAILED(ehr)) continue;

            UINT32 avail = renBufFrames - padding;
            if (avail == 0) continue;

            BYTE* out = nullptr;
            ehr = ren->GetBuffer(avail, &out);
            if (FAILED(ehr)) continue;

            const int outCh = renFmt->nChannels;

            size_t need = (size_t)avail;
            if (tmp.size() < need) tmp.resize(need);

            size_t got = outRing.pop(tmp.data(), need);
            if (got < need) std::fill(tmp.begin() + (ptrdiff_t)got, tmp.begin() + (ptrdiff_t)need, 0.0f);

            if (renFloat) {
                float* f = reinterpret_cast<float*>(out);
                for (UINT32 i = 0; i < avail; i++) {
                    float s = tmp[i];
                    for (int c = 0; c < outCh; c++) f[i * outCh + c] = s;
                }
            }
            else {
                int16_t* p = reinterpret_cast<int16_t*>(out);
                for (UINT32 i = 0; i < avail; i++) {
                    float s = Clamp(tmp[i], -1.0f, 1.0f);
                    int16_t v = (int16_t)std::lrintf(s * 32767.0f);
                    for (int c = 0; c < outCh; c++) p[i * outCh + c] = v;
                }
            }

            ren->ReleaseBuffer(avail, 0);
        }

        if (hTask) AvRevertMmThreadCharacteristics(hTask);
        });

    auto printSelectedParamLocked = [&]() {
        if (proc.params.empty()) return;
        const auto& p = proc.params[proc.paramIndex];
        std::printf("\n[Param] %s = %.3f\n", p.name, *p.value);
    };

    auto printSelectedParam = [&]() {
        std::lock_guard<std::mutex> lock(procMtx);
        printSelectedParamLocked();
    };

    printSelectedParam();

    while (g_running.load()) {
        while (_kbhit()) {
            int ch = _getch();

            if (ch == 'q' || ch == 'Q') { g_running.store(false); break; }

            if (ch == '1') { std::lock_guard<std::mutex> lock(procMtx); proc.setMode(Mode::FM);  std::printf("\n[Mode] FM\n");  printSelectedParamLocked(); }
            if (ch == '2') { std::lock_guard<std::mutex> lock(procMtx); proc.setMode(Mode::AM);  std::printf("\n[Mode] AM\n");  printSelectedParamLocked(); }
            if (ch == '3') { std::lock_guard<std::mutex> lock(procMtx); proc.setMode(Mode::SSB); std::printf("\n[Mode] SSB\n"); printSelectedParamLocked(); }

            if (ch == '	') {
                std::lock_guard<std::mutex> lock(procMtx);
                if (!proc.params.empty()) {
                    proc.paramIndex = (proc.paramIndex + 1) % (int)proc.params.size();
                    printSelectedParamLocked();
                }
            }

            auto adjust = [&](float delta) {
                if (proc.params.empty()) return;
                auto& p = proc.params[proc.paramIndex];
                *p.value = Clamp(*p.value + delta, p.minV, p.maxV);
                proc.applyPresetToDSP();
                printSelectedParamLocked();
            };

            if (ch == '[') { std::lock_guard<std::mutex> lock(procMtx); if (!proc.params.empty()) adjust(-proc.params[proc.paramIndex].step); }
            if (ch == ']') { std::lock_guard<std::mutex> lock(procMtx); if (!proc.params.empty()) adjust(+proc.params[proc.paramIndex].step); }
            if (ch == '{') { std::lock_guard<std::mutex> lock(procMtx); if (!proc.params.empty()) adjust(-proc.params[proc.paramIndex].bigStep); }
            if (ch == '}') { std::lock_guard<std::mutex> lock(procMtx); if (!proc.params.empty()) adjust(+proc.params[proc.paramIndex].bigStep); }

            if (ch == 'g' || ch == 'G') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->gateOn = !proc.cur->gateOn; proc.applyPresetToDSP(); std::printf("\n[Gate] %s\n", proc.cur->gateOn ? "ON" : "OFF"); }
            if (ch == 'e' || ch == 'E') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->eqOn = !proc.cur->eqOn; proc.applyPresetToDSP(); std::printf("\n[EQ] %s\n", proc.cur->eqOn ? "ON" : "OFF"); }
            if (ch == 'p' || ch == 'P') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->plosiveOn = !proc.cur->plosiveOn; proc.applyPresetToDSP(); std::printf("\n[Plosive] %s\n", proc.cur->plosiveOn ? "ON" : "OFF"); }
            if (ch == 'r' || ch == 'R') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->phaseRotOn = !proc.cur->phaseRotOn; proc.applyPresetToDSP(); std::printf("\n[PhaseRot] %s\n", proc.cur->phaseRotOn ? "ON" : "OFF"); }
            if (ch == 'a' || ch == 'A') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->agcOn = !proc.cur->agcOn; proc.applyPresetToDSP(); std::printf("\n[AGC] %s\n", proc.cur->agcOn ? "ON" : "OFF"); }
            if (ch == 'b' || ch == 'B') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->mbOn = !proc.cur->mbOn; proc.applyPresetToDSP(); std::printf("\n[Multiband] %s\n", proc.cur->mbOn ? "ON" : "OFF"); }
            if (ch == 'd' || ch == 'D') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->deessOn = !proc.cur->deessOn; proc.applyPresetToDSP(); std::printf("\n[DeEss] %s\n", proc.cur->deessOn ? "ON" : "OFF"); }
            if (ch == 'x' || ch == 'X') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->excOn = !proc.cur->excOn; proc.applyPresetToDSP(); std::printf("\n[Exciter] %s\n", proc.cur->excOn ? "ON" : "OFF"); }
            if (ch == 'm' || ch == 'M') { std::lock_guard<std::mutex> lock(procMtx); proc.cur->monSim = !proc.cur->monSim; proc.applyPresetToDSP(); std::printf("\n[Monitor] %s\n", proc.cur->monSim ? "ON" : "OFF"); }

            if (ch == 's' || ch == 'S') {
                std::lock_guard<std::mutex> lock(procMtx);
                SaveIni("maul_preset.ini", proc.fm, proc.am, proc.ssb);
                std::printf("\n[Saved] maul_preset.ini\n");
            }
            if (ch == 'l' || ch == 'L') {
                std::lock_guard<std::mutex> lock(procMtx);
                if (LoadIni("maul_preset.ini", proc.fm, proc.am, proc.ssb)) {
                    proc.setMode(proc.mode);
                    std::printf("\n[Loaded] maul_preset.ini\n");
                    printSelectedParamLocked();
                }
                else {
                    std::printf("\n[Load failed] maul_preset.ini\n");
                }
            }

            if (ch == 't' || ch == 'T') {
                int samples = (int)(renRate * 5.0);
                toneSamplesLeft.store(samples);
                std::printf("\n[Tone] 1 kHz @ -12 dBFS for 5 sec. Set your TX input so this is SAFE deviation.\n");
            }
        }

        float dbfs = meterDbfs.load();
        {
            std::lock_guard<std::mutex> lock(procMtx);
            if (!proc.params.empty()) {
                const auto& p = proc.params[proc.paramIndex];
                std::printf("Level %.1f dBFS | Mode %s | %s=%.3f      \r",
                    dbfs,
                    (proc.mode == Mode::FM ? "FM" : (proc.mode == Mode::AM ? "AM" : "SSB")),
                    p.name, *p.value
                );
            }
            else {
                std::printf("Level %.1f dBFS\r", dbfs);
            }
        }
        std::fflush(stdout);
        Sleep(15);
    }

    std::printf("\nStopping...\n");

    capClient->Stop();
    renClient->Stop();

    if (capThread.joinable()) capThread.join();
    if (dspThread.joinable()) dspThread.join();
    if (renThread.joinable()) renThread.join();

    CloseHandle(hCapEvent);
    CloseHandle(hRenEvent);

    CoTaskMemFree(capFmt);
    CoTaskMemFree(renFmt);
    CoUninitialize();
    return 0;
}
