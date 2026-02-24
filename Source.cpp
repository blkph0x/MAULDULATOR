// MAULSOUNDRADIO.cpp
// Native Windows (WASAPI) real-time voice processor: mic -> DSP -> default render
// - Designed to work with VB-Audio Cable (set app output to CABLE Input via Volume Mixer)
// - Modes: FM / AM / SSB
// - Live parameter editing + save/load INI
//
// Build (VS2019/VS2022):
//  - C++20
//  - Link: ole32.lib uuid.lib avrt.lib

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

using Microsoft::WRL::ComPtr;

static constexpr float kPi = 3.14159265358979323846f;

template <typename T>
static inline T Clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline float dbToLin(float db) { return std::pow(10.0f, db / 20.0f); }
static inline float linToDb(float x) { return 20.0f * std::log10(std::max(x, 1e-12f)); }

static volatile LONG g_running = 1;

static BOOL WINAPI CtrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_CLOSE_EVENT || type == CTRL_BREAK_EVENT) {
        InterlockedExchange(&g_running, 0);
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

// -------------------- DSP building blocks --------------------

// RBJ biquad (Direct Form II)
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

        float b0 = (1 - cw) * 0.5f;
        float b1 = (1 - cw);
        float b2 = (1 - cw) * 0.5f;
        float a0 = 1 + alpha;
        float a1 = -2 * cw;
        float a2 = 1 - alpha;

        q.b0 = b0 / a0; q.b1 = b1 / a0; q.b2 = b2 / a0;
        q.a1 = a1 / a0; q.a2 = a2 / a0;
        return q;
    }

    static Biquad HighPass(float fs, float fc, float Q) {
        Biquad q;
        fc = std::max(10.0f, std::min(fc, fs * 0.49f));
        float w0 = 2.0f * kPi * (fc / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);

        float b0 = (1 + cw) * 0.5f;
        float b1 = -(1 + cw);
        float b2 = (1 + cw) * 0.5f;
        float a0 = 1 + alpha;
        float a1 = -2 * cw;
        float a2 = 1 - alpha;

        q.b0 = b0 / a0; q.b1 = b1 / a0; q.b2 = b2 / a0;
        q.a1 = a1 / a0; q.a2 = a2 / a0;
        return q;
    }

    // Peaking EQ
    static Biquad Peaking(float fs, float f0, float Q, float gainDb) {
        Biquad q;
        f0 = std::max(20.0f, std::min(f0, fs * 0.49f));
        float A = std::pow(10.0f, gainDb / 40.0f);
        float w0 = 2.0f * kPi * (f0 / fs);
        float cw = std::cos(w0), sw = std::sin(w0);
        float alpha = sw / (2.0f * Q);

        float b0 = 1 + alpha * A;
        float b1 = -2 * cw;
        float b2 = 1 - alpha * A;
        float a0 = 1 + alpha / A;
        float a1 = -2 * cw;
        float a2 = 1 - alpha / A;

        q.b0 = b0 / a0; q.b1 = b1 / a0; q.b2 = b2 / a0;
        q.a1 = a1 / a0; q.a2 = a2 / a0;
        return q;
    }
};

// Replace coefficients but preserve DF2 state
static inline void UpdateCoeffsPreserveState(Biquad& dst, const Biquad& src) {
    float z1 = dst.z1, z2 = dst.z2;
    dst = src;
    dst.z1 = z1; dst.z2 = z2;
}

// Envelope follower
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

// Downward expander / gate (simple)
struct GateExpander {
    float fs = 48000.0f;
    float threshDb = -55.0f; // below this, attenuate
    float ratio = 4.0f;      // 1=no gate, higher=stronger
    float floorDb = -40.0f;  // max attenuation (negative)
    bool enabled = true;

    EnvFollower env;
    EnvFollower gainSmooth;

    void init(float Fs) {
        fs = Fs;
        env.set(fs, 3.0f, 120.0f);
        gainSmooth.set(fs, 5.0f, 120.0f);
        env.env = 0; gainSmooth.env = 0;
    }

    float process(float x) {
        if (!enabled) return x;

        float e = env.process(std::fabs(x));
        float eDb = linToDb(e);
        float gainDb = 0.0f;

        if (eDb < threshDb) {
            float under = threshDb - eDb; // positive
            float att = under * (std::max(ratio, 1.0f) - 1.0f);
            float maxAtt = std::max(0.0f, -floorDb); // floorDb is negative
            gainDb = -std::min(att, maxAtt);
        }

        float target = dbToLin(gainDb);
        float smooth = gainSmooth.process(target);
        return x * smooth;
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

    EnvFollower env;
    EnvFollower gainSmooth;

    void init(float Fs, float attackMs, float releaseMs, float gAttackMs, float gReleaseMs) {
        fs = Fs;
        env.set(fs, attackMs, releaseMs);
        gainSmooth.set(fs, gAttackMs, gReleaseMs);
        env.env = 0; gainSmooth.env = 0;
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

// Lookahead limiter (peak)
struct LookaheadLimiter {
    float fs = 48000.0f;
    float ceilingDb = -1.0f;
    float ceiling = 0.891f;
    float attackMs = 1.0f;
    float releaseMs = 60.0f;
    float lookaheadMs = 6.0f;
    bool enabled = true;

    int win = 0;
    int sampleIndex = 0;
    float g = 1.0f;

    float aA = 0, aR = 0;
    std::deque<float> buf;
    MaxQueue mq;

    void init(float Fs) {
        fs = Fs;
        win = std::max(8, (int)std::round(fs * (lookaheadMs / 1000.0f)));
        buf.clear();
        mq.clear();
        sampleIndex = 0;
        g = 1.0f;
        aA = std::exp(-1.0f / (fs * (attackMs / 1000.0f)));
        aR = std::exp(-1.0f / (fs * (releaseMs / 1000.0f)));
        setCeilingDb(ceilingDb);
    }

    void setCeilingDb(float db) { ceilingDb = db; ceiling = dbToLin(db); }

    float process(float x) {
        if (!enabled) return x;

        buf.push_back(x);
        mq.push(std::fabs(x));
        sampleIndex++;

        if ((int)buf.size() < win) {
            // Warmup: output 0 for only lookahead duration.
            return 0.0f;
        }

        int oldestIdx = sampleIndex - (int)buf.size();
        mq.popOld(oldestIdx);
        float peak = mq.max();

        float target = 1.0f;
        if (peak > ceiling && peak > 1e-9f) target = ceiling / peak;

        if (target < g) g = aA * g + (1.0f - aA) * target;
        else            g = aR * g + (1.0f - aR) * target;

        float y = buf.front();
        buf.pop_front();
        return y * g;
    }
};

// De-esser: detect sibilant band and attenuate it
struct DeEsser {
    float fs = 48000.0f;
    bool enabled = true;

    float detAmount = 0.6f;
    float threshDb = -30.0f;
    float maxRedDb = 10.0f;

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

    float process(float x) {
        if (!enabled) return x;

        float s = hp2.process(hp1.process(x));
        s = lp2.process(lp1.process(s));
        float e = env.process(std::fabs(s));
        float eDb = linToDb(e);

        float redDb = 0.0f;
        if (eDb > threshDb) {
            float over = eDb - threshDb;
            redDb = std::min(over * 1.5f, maxRedDb);
        }

        float hfGain = dbToLin(-redDb * detAmount);
        return x - (1.0f - hfGain) * s;
    }
};

// Presence exciter: distort only 1.5k-3.5k band then add harmonics back
struct Exciter {
    float fs = 48000.0f;
    bool enabled = true;
    float amount = 0.35f; // 0..1
    float drive = 2.2f;   // 1..6

    Biquad hp1, hp2, lp1, lp2;

    void init(float Fs) {
        fs = Fs;
        const float Q1 = 0.5411961f, Q2 = 1.3065630f;
        hp1 = Biquad::HighPass(fs, 1500.0f, Q1);
        hp2 = Biquad::HighPass(fs, 1500.0f, Q2);
        lp1 = Biquad::LowPass(fs, 3500.0f, Q1);
        lp2 = Biquad::LowPass(fs, 3500.0f, Q2);
        hp1.reset(); hp2.reset(); lp1.reset(); lp2.reset();
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

// Monitor speaker sim (for headphones) ~300-3.2k + honk
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

// Linear resampler (capture rate -> render rate)
struct LinearResampler {
    double inRate = 48000.0, outRate = 48000.0;
    double step = 1.0;
    double pos = 0.0;

    void setRates(double inR, double outR) {
        inRate = inR; outRate = outR;
        step = inRate / outRate;
        pos = 0.0;
    }

    void produce(std::deque<float>& inBuf, std::vector<float>& out) {
        out.clear();
        if (inBuf.size() < 2) return;

        while ((pos + 1.0) < (double)inBuf.size()) {
            int i = (int)pos;
            double frac = pos - (double)i;
            float s0 = inBuf[(size_t)i];
            float s1 = inBuf[(size_t)i + 1];
            out.push_back((float)(s0 + (s1 - s0) * frac));
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

// -------------------- Main processor (mode + chain) --------------------

enum class Mode { FM = 1, AM = 2, SSB = 3 };

struct Preset {
    // bandlimits
    float hpHz = 120.0f;
    float lpHz = 3200.0f;

    // EQ (3 bands)
    bool  eqOn = true;
    float eqLowDb = +2.5f;   // ~220 Hz
    float eqNasalDb = -3.0f;   // ~800 Hz
    float eqPresDb = +4.0f;   // ~2400 Hz

    // gate
    bool  gateOn = true;
    float gateThreshDb = -55.0f;
    float gateRatio = 4.0f;
    float gateFloorDb = -35.0f;

    // compressors
    bool  comp1On = true, comp2On = true;
    float c1Thresh = -26.0f, c1Ratio = 5.0f, c1Makeup = 6.0f;
    float c2Thresh = -18.0f, c2Ratio = 12.0f, c2Makeup = 1.0f;

    // de-esser
    bool  deessOn = true;
    float deessThreshDb = -30.0f;
    float deessAmt = 0.6f;

    // exciter
    bool  excOn = true;
    float excAmt = 0.35f;
    float excDrive = 2.2f;

    // limiter
    bool  limOn = true;
    float limCeilDb = -1.0f;
    float limLookaheadMs = 6.0f;
    float limReleaseMs = 60.0f;

    // clipper style (mode-specific)
    float amPosCeil = 0.98f;
    float amNegCeil = 0.55f;
    float satDrive = 1.6f;

    // output
    float outGainDb = 0.0f;

    // monitor
    bool monSim = false;
};

// Parameter UI
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

    // bandlimit filters
    Biquad hpf;
    Biquad lpf1, lpf2;

    // eq filters
    Biquad eqLow, eqNasal, eqPres;

    // modules
    GateExpander gate;
    Compressor comp1, comp2;
    DeEsser deess;
    Exciter exc;
    LookaheadLimiter lim;
    MonitorSim mon;

    // pre-emphasis style bite (simple differentiator)
    float preAlpha = 0.88f;
    float prevX = 0.0f;

    // output gain cached
    float outGainLin = 1.0f;

    // parameter list
    std::vector<ParamDesc> params;
    int paramIndex = 0;

    // dirty sync
    bool dirty = true;

    // cached last values (to update DSP without wiping state)
    float lastHpHz = 1e9f;
    float lastLpHz = 1e9f;

    float lastEqLowDb = 1e9f;
    float lastEqNasalDb = 1e9f;
    float lastEqPresDb = 1e9f;
    bool  lastEqOn = false;

    float lastLimCeilDb = 1e9f;
    float lastLimLookMs = 1e9f;
    float lastLimRelMs = 1e9f;
    bool  lastLimOn = false;

    bool  lastMonSim = false;

    Processor() {
        // FM defaults
        fm = Preset{};
        fm.hpHz = 140; fm.lpHz = 3000;
        fm.eqPresDb = +2.5f;
        fm.c1Thresh = -28; fm.c1Ratio = 4; fm.c1Makeup = 4;
        fm.c2Thresh = -18; fm.c2Ratio = 10; fm.c2Makeup = 0;
        fm.excAmt = 0.18f; fm.excDrive = 1.8f;
        fm.limCeilDb = -3.0f;
        fm.satDrive = 1.15f;
        fm.deessOn = true;
        fm.monSim = false;

        // AM defaults
        am = Preset{};
        am.hpHz = 85;  am.lpHz = 4200;
        am.eqLowDb = +3.0f; am.eqNasalDb = -3.0f; am.eqPresDb = +5.0f;
        am.c1Thresh = -26; am.c1Ratio = 5;  am.c1Makeup = 6;
        am.c2Thresh = -18; am.c2Ratio = 12; am.c2Makeup = 2;
        am.excAmt = 0.35f; am.excDrive = 2.4f;
        am.deessOn = false;
        am.limCeilDb = -1.0f;
        am.amPosCeil = 0.98f; am.amNegCeil = 0.55f;
        am.satDrive = 1.9f;
        am.monSim = true; // headphone-only; disable if feeding TX chain

        // SSB defaults
        ssb = Preset{};
        ssb.hpHz = 180; ssb.lpHz = 2900;
        ssb.eqPresDb = +4.0f;
        ssb.c1Thresh = -26; ssb.c1Ratio = 6;  ssb.c1Makeup = 5;
        ssb.c2Thresh = -18; ssb.c2Ratio = 10; ssb.c2Makeup = 1;
        ssb.excAmt = 0.22f; ssb.excDrive = 2.0f;
        ssb.deessOn = true;
        ssb.limCeilDb = -1.0f;
        ssb.satDrive = 1.35f;
        ssb.monSim = false;
    }

    void init(float Fs, Mode m) {
        fs = Fs;
        setMode(m);
    }

    void setMode(Mode m) {
        mode = m;
        if (m == Mode::FM) cur = &fm;
        else if (m == Mode::AM) cur = &am;
        else cur = &ssb;

        // Reset cached values so everything is rebuilt cleanly for this mode
        lastHpHz = lastLpHz = 1e9f;
        lastEqLowDb = lastEqNasalDb = lastEqPresDb = 1e9f;
        lastEqOn = !cur->eqOn;

        lastLimCeilDb = lastLimLookMs = lastLimRelMs = 1e9f;
        lastLimOn = !cur->limOn;

        lastMonSim = !cur->monSim;

        // Init modules (this resets their internal state)
        gate.init(fs);
        comp1.init(fs, 4.0f, 220.0f, 5.0f, 90.0f);
        comp2.init(fs, 1.0f, 90.0f, 3.0f, 70.0f);

        deess.init(fs);
        exc.init(fs);
        mon.init(fs);

        lim.attackMs = 1.0f;
        lim.lookaheadMs = cur->limLookaheadMs;
        lim.releaseMs = cur->limReleaseMs;
        lim.setCeilingDb(cur->limCeilDb);
        lim.init(fs);

        // pre-emphasis strength
        preAlpha = (mode == Mode::AM) ? 0.90f : (mode == Mode::SSB ? 0.86f : 0.84f);
        prevX = 0.0f;

        rebuildParamList();

        dirty = true;
        syncFromPreset(); // build filters/vars immediately
    }

    void rebuildParamList() {
        params.clear();

        params.push_back({ "outGain(dB)", &cur->outGainDb, 1.0f, 4.0f, -30.0f, +24.0f });
        params.push_back({ "HP(Hz)", &cur->hpHz, 10.0f, 50.0f, 40.0f, 400.0f });
        params.push_back({ "LP(Hz)", &cur->lpHz, 50.0f, 250.0f, 2000.0f, 6000.0f });

        params.push_back({ "EQ low(dB)", &cur->eqLowDb, 0.5f, 2.0f, -12.0f, +12.0f });
        params.push_back({ "EQ nasal(dB)", &cur->eqNasalDb, 0.5f, 2.0f, -12.0f, +12.0f });
        params.push_back({ "EQ pres(dB)", &cur->eqPresDb, 0.5f, 2.0f, -12.0f, +12.0f });

        params.push_back({ "Gate thr(dB)", &cur->gateThreshDb, 1.0f, 4.0f, -80.0f, -20.0f });
        params.push_back({ "Gate ratio", &cur->gateRatio, 0.5f, 2.0f, 1.0f, 12.0f });
        params.push_back({ "Gate floor(dB)", &cur->gateFloorDb, 1.0f, 4.0f, -60.0f, 0.0f });

        params.push_back({ "C1 thr(dB)", &cur->c1Thresh, 1.0f, 4.0f, -60.0f, 0.0f });
        params.push_back({ "C1 ratio", &cur->c1Ratio, 0.5f, 2.0f, 1.0f, 20.0f });
        params.push_back({ "C1 makeup(dB)", &cur->c1Makeup, 0.5f, 2.0f, -6.0f, +18.0f });

        params.push_back({ "C2 thr(dB)", &cur->c2Thresh, 1.0f, 4.0f, -60.0f, 0.0f });
        params.push_back({ "C2 ratio", &cur->c2Ratio, 0.5f, 2.0f, 1.0f, 30.0f });
        params.push_back({ "C2 makeup(dB)", &cur->c2Makeup, 0.5f, 2.0f, -6.0f, +18.0f });

        params.push_back({ "DeEss thr(dB)", &cur->deessThreshDb, 1.0f, 4.0f, -60.0f, -10.0f });
        params.push_back({ "DeEss amt", &cur->deessAmt, 0.05f, 0.2f, 0.0f, 1.0f });

        params.push_back({ "Exc amt", &cur->excAmt, 0.05f, 0.2f, 0.0f, 1.0f });
        params.push_back({ "Exc drive", &cur->excDrive, 0.2f, 0.8f, 1.0f, 6.0f });

        params.push_back({ "Lim ceil(dB)", &cur->limCeilDb, 0.5f, 2.0f, -12.0f, 0.0f });
        params.push_back({ "Lim look(ms)", &cur->limLookaheadMs, 1.0f, 3.0f, 2.0f, 20.0f });
        params.push_back({ "Lim rel(ms)", &cur->limReleaseMs, 5.0f, 20.0f, 20.0f, 300.0f });

        params.push_back({ "AM +ceil", &cur->amPosCeil, 0.01f, 0.05f, 0.50f, 1.00f });
        params.push_back({ "AM -ceil", &cur->amNegCeil, 0.01f, 0.05f, 0.20f, 1.00f });
        params.push_back({ "Sat drive", &cur->satDrive, 0.10f, 0.40f, 1.00f, 4.00f });

        paramIndex = std::min(paramIndex, (int)params.size() - 1);
    }

    void markDirty() { dirty = true; }

    // Sync DSP objects from current preset.
    // This updates coefficients/params without wiping filter state unnecessarily.
    void syncFromPreset() {
        if (!dirty || !cur) return;
        dirty = false;

        // Bandlimits (cutoff changes -> rebuild + reset those filters to avoid weird transient state)
        if (cur->hpHz != lastHpHz) {
            hpf = Biquad::HighPass(fs, cur->hpHz, 0.7071f);
            hpf.reset();
            lastHpHz = cur->hpHz;
        }
        if (cur->lpHz != lastLpHz) {
            const float Q1 = 0.5411961f, Q2 = 1.3065630f;
            lpf1 = Biquad::LowPass(fs, cur->lpHz, Q1);
            lpf2 = Biquad::LowPass(fs, cur->lpHz, Q2);
            lpf1.reset(); lpf2.reset();
            lastLpHz = cur->lpHz;
        }

        // EQ coefficients (preserve state; do NOT recreate biquads per-sample)
        if (cur->eqOn != lastEqOn) {
            // When toggling EQ, keep states but ensure coefficients match current values
            UpdateCoeffsPreserveState(eqLow, Biquad::Peaking(fs, 220.0f, 0.9f, cur->eqLowDb));
            UpdateCoeffsPreserveState(eqNasal, Biquad::Peaking(fs, 800.0f, 1.2f, cur->eqNasalDb));
            UpdateCoeffsPreserveState(eqPres, Biquad::Peaking(fs, 2400.0f, 1.0f, cur->eqPresDb));
            lastEqLowDb = cur->eqLowDb;
            lastEqNasalDb = cur->eqNasalDb;
            lastEqPresDb = cur->eqPresDb;
            lastEqOn = cur->eqOn;
        }
        else {
            if (cur->eqLowDb != lastEqLowDb) {
                UpdateCoeffsPreserveState(eqLow, Biquad::Peaking(fs, 220.0f, 0.9f, cur->eqLowDb));
                lastEqLowDb = cur->eqLowDb;
            }
            if (cur->eqNasalDb != lastEqNasalDb) {
                UpdateCoeffsPreserveState(eqNasal, Biquad::Peaking(fs, 800.0f, 1.2f, cur->eqNasalDb));
                lastEqNasalDb = cur->eqNasalDb;
            }
            if (cur->eqPresDb != lastEqPresDb) {
                UpdateCoeffsPreserveState(eqPres, Biquad::Peaking(fs, 2400.0f, 1.0f, cur->eqPresDb));
                lastEqPresDb = cur->eqPresDb;
            }
        }

        // Gate parameters
        gate.enabled = cur->gateOn;
        gate.threshDb = cur->gateThreshDb;
        gate.ratio = std::max(1.0f, cur->gateRatio);
        gate.floorDb = cur->gateFloorDb;

        // Compressor parameters
        comp1.enabled = cur->comp1On;
        comp1.threshDb = cur->c1Thresh;
        comp1.ratio = std::max(1.0f, cur->c1Ratio);
        comp1.setMakeupDb(cur->c1Makeup);

        comp2.enabled = cur->comp2On;
        comp2.threshDb = cur->c2Thresh;
        comp2.ratio = std::max(1.0f, cur->c2Ratio);
        comp2.setMakeupDb(cur->c2Makeup);

        // De-esser parameters
        deess.enabled = cur->deessOn;
        deess.threshDb = cur->deessThreshDb;
        deess.detAmount = cur->deessAmt;

        // Exciter parameters
        exc.enabled = cur->excOn;
        exc.amount = cur->excAmt;
        exc.drive = cur->excDrive;

        // Monitor sim
        mon.enabled = cur->monSim;

        // Output gain
        outGainLin = dbToLin(cur->outGainDb);

        // Limiter parameters: if lookahead or release changed, re-init (so win + smoothing update)
        if (cur->limOn != lastLimOn ||
            cur->limLookaheadMs != lastLimLookMs ||
            cur->limReleaseMs != lastLimRelMs) {
            lim.enabled = cur->limOn;
            lim.lookaheadMs = cur->limLookaheadMs;
            lim.releaseMs = cur->limReleaseMs;
            lim.init(fs); // resets limiter buffer; expected for lookahead changes
            lastLimOn = cur->limOn;
            lastLimLookMs = cur->limLookaheadMs;
            lastLimRelMs = cur->limReleaseMs;
            lastLimCeilDb = 1e9f; // force ceiling refresh below
        }
        else {
            lim.enabled = cur->limOn;
        }

        if (cur->limCeilDb != lastLimCeilDb) {
            lim.setCeilingDb(cur->limCeilDb);
            lastLimCeilDb = cur->limCeilDb;
        }

        // track mon flag cache (not strictly needed here; kept if you extend later)
        lastMonSim = cur->monSim;
    }

    float satSym(float x, float d) const {
        d = std::max(1.0f, d);
        // NOTE: this mapping can expand small signals; limiter is placed LAST to enforce ceiling.
        return std::tanh(x * d) / std::tanh(d);
    }

    float clipAM(float x) const {
        float pos = Clamp(cur->amPosCeil, 0.5f, 1.0f);
        float neg = Clamp(cur->amNegCeil, 0.2f, 1.0f);
        x = Clamp(x, -neg, pos);
        return satSym(x, cur->satDrive);
    }

    float processOne(float x) {
        // Assume syncFromPreset() already called before processing blocks.

        // Band-limit first
        float y = hpf.process(x);
        y = lpf2.process(lpf1.process(y));

        // Gate/expander
        y = gate.process(y);

        // EQ
        if (cur->eqOn) {
            y = eqPres.process(eqNasal.process(eqLow.process(y)));
        }

        // mild pre-emphasis bite (simple differentiator)
        float pe = y - preAlpha * prevX;
        prevX = y;

        // Compression (2-stage)
        pe = comp1.process(pe);
        pe = comp2.process(pe);

        // Exciter
        pe = exc.process(pe);

        // De-esser
        pe = deess.process(pe);

        // Mode character (saturation / AM asym clip)
        if (mode == Mode::AM) {
            pe = clipAM(pe);
        }
        else {
            pe = satSym(Clamp(pe, -1.0f, 1.0f), cur->satDrive);
        }

        // Monitor sim (headphones)
        pe = mon.process(pe);

        // Output gain (pre-limiter)
        pe *= outGainLin;

        // LIMITER LAST so ceiling is real, even after saturation/monitor filtering/outGain.
        pe = lim.process(pe);

        return pe;
    }
};

// -------------------- INI save/load --------------------

static void SaveIni(const char* path, const Preset& fm, const Preset& am, const Preset& ssb) {
    auto writePreset = [&](std::ofstream& o, const char* name, const Preset& p) {
        o << "[" << name << "]\n";
        o << "hpHz=" << p.hpHz << "\n";
        o << "lpHz=" << p.lpHz << "\n";
        o << "eqOn=" << (p.eqOn ? 1 : 0) << "\n";
        o << "eqLowDb=" << p.eqLowDb << "\n";
        o << "eqNasalDb=" << p.eqNasalDb << "\n";
        o << "eqPresDb=" << p.eqPresDb << "\n";

        o << "gateOn=" << (p.gateOn ? 1 : 0) << "\n";
        o << "gateThreshDb=" << p.gateThreshDb << "\n";
        o << "gateRatio=" << p.gateRatio << "\n";
        o << "gateFloorDb=" << p.gateFloorDb << "\n";

        o << "comp1On=" << (p.comp1On ? 1 : 0) << "\n";
        o << "c1Thresh=" << p.c1Thresh << "\n";
        o << "c1Ratio=" << p.c1Ratio << "\n";
        o << "c1Makeup=" << p.c1Makeup << "\n";

        o << "comp2On=" << (p.comp2On ? 1 : 0) << "\n";
        o << "c2Thresh=" << p.c2Thresh << "\n";
        o << "c2Ratio=" << p.c2Ratio << "\n";
        o << "c2Makeup=" << p.c2Makeup << "\n";

        o << "deessOn=" << (p.deessOn ? 1 : 0) << "\n";
        o << "deessThreshDb=" << p.deessThreshDb << "\n";
        o << "deessAmt=" << p.deessAmt << "\n";

        o << "excOn=" << (p.excOn ? 1 : 0) << "\n";
        o << "excAmt=" << p.excAmt << "\n";
        o << "excDrive=" << p.excDrive << "\n";

        o << "limOn=" << (p.limOn ? 1 : 0) << "\n";
        o << "limCeilDb=" << p.limCeilDb << "\n";
        o << "limLookaheadMs=" << p.limLookaheadMs << "\n";
        o << "limReleaseMs=" << p.limReleaseMs << "\n";

        o << "amPosCeil=" << p.amPosCeil << "\n";
        o << "amNegCeil=" << p.amNegCeil << "\n";
        o << "satDrive=" << p.satDrive << "\n";

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
    auto* cur = (Preset*)nullptr;

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

        if (k == "hpHz") cur->hpHz = f();
        else if (k == "lpHz") cur->lpHz = f();
        else if (k == "eqOn") cur->eqOn = ParseBool(v);
        else if (k == "eqLowDb") cur->eqLowDb = f();
        else if (k == "eqNasalDb") cur->eqNasalDb = f();
        else if (k == "eqPresDb") cur->eqPresDb = f();

        else if (k == "gateOn") cur->gateOn = ParseBool(v);
        else if (k == "gateThreshDb") cur->gateThreshDb = f();
        else if (k == "gateRatio") cur->gateRatio = f();
        else if (k == "gateFloorDb") cur->gateFloorDb = f();

        else if (k == "comp1On") cur->comp1On = ParseBool(v);
        else if (k == "c1Thresh") cur->c1Thresh = f();
        else if (k == "c1Ratio") cur->c1Ratio = f();
        else if (k == "c1Makeup") cur->c1Makeup = f();

        else if (k == "comp2On") cur->comp2On = ParseBool(v);
        else if (k == "c2Thresh") cur->c2Thresh = f();
        else if (k == "c2Ratio") cur->c2Ratio = f();
        else if (k == "c2Makeup") cur->c2Makeup = f();

        else if (k == "deessOn") cur->deessOn = ParseBool(v);
        else if (k == "deessThreshDb") cur->deessThreshDb = f();
        else if (k == "deessAmt") cur->deessAmt = f();

        else if (k == "excOn") cur->excOn = ParseBool(v);
        else if (k == "excAmt") cur->excAmt = f();
        else if (k == "excDrive") cur->excDrive = f();

        else if (k == "limOn") cur->limOn = ParseBool(v);
        else if (k == "limCeilDb") cur->limCeilDb = f();
        else if (k == "limLookaheadMs") cur->limLookaheadMs = f();
        else if (k == "limReleaseMs") cur->limReleaseMs = f();

        else if (k == "amPosCeil") cur->amPosCeil = f();
        else if (k == "amNegCeil") cur->amNegCeil = f();
        else if (k == "satDrive") cur->satDrive = f();

        else if (k == "outGainDb") cur->outGainDb = f();
        else if (k == "monSim") cur->monSim = ParseBool(v);
    }

    return true;
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

    // Prefer console defaults first, then communications/multimedia.
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

    std::printf("\nHotkeys: 1=FM 2=AM 3=SSB | TAB select param | [ ] adjust | { } big adjust\n");
    std::printf("Toggles: G gate | E EQ | D de-ess | X exciter | M monitorSim | S save | L load | Q quit\n");
    std::printf("Tip: If feeding a TX chain (VB-Cable -> SDR), consider turning MonitorSim OFF (press M).\n\n");

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

    // MMCSS
    DWORD taskIndex = 0;
    HANDLE hTask = AvSetMmThreadCharacteristicsW(L"Pro Audio", &taskIndex);

    Processor proc;
    proc.init((float)renRate, Mode::AM);

    LinearResampler rs;
    rs.setRates(capRate, renRate);

    std::deque<float> capMono;
    std::deque<float> play;
    std::vector<float> rsOut;
    rsOut.reserve(4096);

    // meter
    double meterSumSq = 0.0;
    uint64_t meterCount = 0;
    ULONGLONG lastMeterMs = GetTickCount64();

    hr = capClient->Start();
    if (FAILED(hr)) Die("capClient->Start failed", hr);
    hr = renClient->Start();
    if (FAILED(hr)) Die("renClient->Start failed", hr);

    HANDLE handles[2] = { hCapEvent, hRenEvent };

    auto printSelectedParam = [&]() {
        if (proc.params.empty()) return;
        const auto& p = proc.params[proc.paramIndex];
        std::printf("\n[Param] %s = %.3f\n", p.name, *p.value);
    };

    printSelectedParam();

    auto handleCapture = [&]() {
        UINT32 packet = 0;
        hr = cap->GetNextPacketSize(&packet);
        if (FAILED(hr)) return;

        while (packet > 0) {
            BYTE* data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;
            hr = cap->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
            if (FAILED(hr)) break;

            const int inCh = capFmt->nChannels;

            if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                for (UINT32 i = 0; i < frames; i++) capMono.push_back(0.0f);
            }
            else if (capFloat) {
                const float* f = reinterpret_cast<const float*>(data);
                for (UINT32 i = 0; i < frames; i++) {
                    float s = 0.0f;
                    for (int c = 0; c < inCh; c++) s += f[i * inCh + c];
                    capMono.push_back(s / (float)inCh);
                }
            }
            else {
                const int16_t* p = reinterpret_cast<const int16_t*>(data);
                for (UINT32 i = 0; i < frames; i++) {
                    float s = 0.0f;
                    for (int c = 0; c < inCh; c++) s += (float)p[i * inCh + c] / 32768.0f;
                    capMono.push_back(s / (float)inCh);
                }
            }

            cap->ReleaseBuffer(frames);
            hr = cap->GetNextPacketSize(&packet);
            if (FAILED(hr)) break;
        }

        // Resample -> process -> enqueue
        rs.produce(capMono, rsOut);

        // Apply parameter changes once per block (not per sample)
        proc.syncFromPreset();

        for (float x : rsOut) {
            float y = proc.processOne(x);
            play.push_back(y);
            meterSumSq += (double)y * (double)y;
            meterCount++;
        }

        // Prevent runaway latency if render can't keep up:
        const size_t maxQueue = (size_t)renBufFrames * 6; // ~60ms at 10ms buffers
        while (play.size() > maxQueue) play.pop_front();
    };

    auto handleRender = [&]() {
        UINT32 padding = 0;
        hr = renClient->GetCurrentPadding(&padding);
        if (FAILED(hr)) return;

        UINT32 avail = renBufFrames - padding;
        if (avail == 0) return;

        BYTE* out = nullptr;
        hr = ren->GetBuffer(avail, &out);
        if (FAILED(hr)) return;

        const int outCh = renFmt->nChannels;

        if (renFloat) {
            float* f = reinterpret_cast<float*>(out);
            for (UINT32 i = 0; i < avail; i++) {
                float s = 0.0f;
                if (!play.empty()) { s = play.front(); play.pop_front(); }
                for (int c = 0; c < outCh; c++) f[i * outCh + c] = s;
            }
        }
        else {
            int16_t* p = reinterpret_cast<int16_t*>(out);
            for (UINT32 i = 0; i < avail; i++) {
                float s = 0.0f;
                if (!play.empty()) { s = play.front(); play.pop_front(); }
                s = Clamp(s, -1.0f, 1.0f);
                int16_t v = (int16_t)std::lrintf(s * 32767.0f);
                for (int c = 0; c < outCh; c++) p[i * outCh + c] = v;
            }
        }

        ren->ReleaseBuffer(avail, 0);
    };

    while (InterlockedCompareExchange(&g_running, 1, 1)) {
        // Hotkeys
        while (_kbhit()) {
            int ch = _getch();

            if (ch == 'q' || ch == 'Q') { InterlockedExchange(&g_running, 0); break; }

            if (ch == '1') { proc.setMode(Mode::FM);  std::printf("\n[Mode] FM\n"); printSelectedParam(); }
            if (ch == '2') { proc.setMode(Mode::AM);  std::printf("\n[Mode] AM\n"); printSelectedParam(); }
            if (ch == '3') { proc.setMode(Mode::SSB); std::printf("\n[Mode] SSB\n"); printSelectedParam(); }

            if (ch == '\t') {
                proc.paramIndex = (proc.paramIndex + 1) % (int)proc.params.size();
                printSelectedParam();
            }

            auto adjust = [&](float delta) {
                if (proc.params.empty()) return;
                auto& p = proc.params[proc.paramIndex];
                *p.value = Clamp(*p.value + delta, p.minV, p.maxV);
                proc.markDirty();
                printSelectedParam();
            };

            if (ch == '[') adjust(-proc.params[proc.paramIndex].step);
            if (ch == ']') adjust(+proc.params[proc.paramIndex].step);

            if (ch == '{') adjust(-proc.params[proc.paramIndex].bigStep);
            if (ch == '}') adjust(+proc.params[proc.paramIndex].bigStep);

            // toggles
            if (ch == 'g' || ch == 'G') { proc.cur->gateOn = !proc.cur->gateOn; proc.markDirty(); std::printf("\n[Gate] %s\n", proc.cur->gateOn ? "ON" : "OFF"); }
            if (ch == 'e' || ch == 'E') { proc.cur->eqOn = !proc.cur->eqOn;   proc.markDirty(); std::printf("\n[EQ] %s\n", proc.cur->eqOn ? "ON" : "OFF"); }
            if (ch == 'd' || ch == 'D') { proc.cur->deessOn = !proc.cur->deessOn; proc.markDirty(); std::printf("\n[DeEss] %s\n", proc.cur->deessOn ? "ON" : "OFF"); }
            if (ch == 'x' || ch == 'X') { proc.cur->excOn = !proc.cur->excOn;  proc.markDirty(); std::printf("\n[Exciter] %s\n", proc.cur->excOn ? "ON" : "OFF"); }
            if (ch == 'm' || ch == 'M') { proc.cur->monSim = !proc.cur->monSim; proc.markDirty(); std::printf("\n[MonitorSim] %s\n", proc.cur->monSim ? "ON" : "OFF"); }

            if (ch == 's' || ch == 'S') {
                SaveIni("maul_preset.ini", proc.fm, proc.am, proc.ssb);
                std::printf("\n[Saved] maul_preset.ini\n");
            }
            if (ch == 'l' || ch == 'L') {
                if (LoadIni("maul_preset.ini", proc.fm, proc.am, proc.ssb)) {
                    // Re-apply current mode (so cur points to correct preset struct)
                    proc.setMode(proc.mode);
                    std::printf("\n[Loaded] maul_preset.ini\n");
                    printSelectedParam();
                }
                else {
                    std::printf("\n[Load failed] maul_preset.ini\n");
                }
            }
        }

        DWORD w = WaitForMultipleObjects(2, handles, FALSE, 50);

        if (w == WAIT_OBJECT_0) {
            handleCapture();
            // also service render immediately if already signaled to reduce jitter
            if (WaitForSingleObject(hRenEvent, 0) == WAIT_OBJECT_0) handleRender();
        }
        else if (w == WAIT_OBJECT_0 + 1) {
            handleRender();
            if (WaitForSingleObject(hCapEvent, 0) == WAIT_OBJECT_0) handleCapture();
        }

        // Meter print
        ULONGLONG now = GetTickCount64();
        if (now - lastMeterMs >= 1000) {
            float rms = (meterCount > 0) ? (float)std::sqrt(meterSumSq / (double)meterCount) : 0.0f;
            float dbfs = linToDb(rms);
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
            std::fflush(stdout);
            meterSumSq = 0.0; meterCount = 0; lastMeterMs = now;
        }
    }

    std::printf("\nStopping...\n");
    capClient->Stop();
    renClient->Stop();

    if (hTask) AvRevertMmThreadCharacteristics(hTask);
    CloseHandle(hCapEvent);
    CloseHandle(hRenEvent);

    CoTaskMemFree(capFmt);
    CoTaskMemFree(renFmt);
    CoUninitialize();
    return 0;
}