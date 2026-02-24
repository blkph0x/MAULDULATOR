# MAULDULATOR
software MAULDULATOR for big CB sound
# MAULSOUNDRADIO 🎙️📻  
Real-time Windows (WASAPI) voice processor: **Mic → DSP → Output device** (speakers/headphones or VB-Audio Cable)

This app takes your **default Windows input (mic)**, applies a “big station / radio” DSP chain in real time, and sends the processed audio to your **default Windows output** (or to **VB-Audio Virtual Cable** if you route it via the Volume Mixer).

It’s designed for:
- **SDR / TX audio chains** (AirSpy / SDR# / other software that can use VB-Cable as a mic source)
- **Voice processing** for streaming / monitoring
- Quick switching between **FM / AM / SSB** voicings

---

## Features

- **WASAPI shared mode** capture + render using event callbacks (low latency).
- Three modes:
  - **FM**: tight, punchy broadcast voice
  - **AM**: big, loud, aggressive “station” sound
  - **SSB**: intelligible comms voice with controlled bandwidth
- Live parameter editing from the keyboard (no GUI needed)
- Save/load presets via `maul_preset.ini`

---

## How audio flows

1. **Capture**: reads audio from the default Windows **Input** device (mic).
2. **Resample**: converts capture sample rate → render sample rate if needed.
3. **DSP chain**: processes audio sample-by-sample.
4. **Render**: writes processed audio to the default Windows **Output** device.

> ✅ Tip: You can route only this app’s output to **VB-Audio Cable** using Windows Volume Mixer (recommended), without changing your whole system output device.

---

## Using with VB-Audio Virtual Cable (recommended for SDR / “use as mic”)

### Goal
Make other apps (AirSpy, SDR software, etc.) receive your processed audio as if it were a microphone.

### Steps (Windows)
1. Install **VB-Audio Virtual Cable**
2. Run `MAULSOUNDRADIO.exe`
3. Open:
   - **Settings → System → Sound → Volume mixer**
4. Find `MAULSOUNDRADIO.exe` and set its **Output device** to:
   - **CABLE Input (VB-Audio Virtual Cable)**
5. In your SDR app (AirSpy/SDR software), select input/mic as:
   - **CABLE Output (VB-Audio Virtual Cable)**

That’s it: the SDR app now “hears” your processed voice.

---

## Controls (hotkeys)

### Mode selection
- `1` = **FM**
- `2` = **AM**
- `3` = **SSB**

### Parameter editing
- `TAB` = cycle through parameters
- `[` = decrease current parameter (small step)
- `]` = increase current parameter (small step)
- `{` = decrease current parameter (big step)
- `}` = increase current parameter (big step)

### Module toggles
- `G` = Gate ON/OFF
- `E` = EQ ON/OFF
- `D` = De-esser ON/OFF
- `X` = Exciter ON/OFF
- `M` = MonitorSim ON/OFF

### Presets
- `S` = save presets to `maul_preset.ini`
- `L` = load presets from `maul_preset.ini`

### Quit
- `Q` = quit

---

## Presets / INI format

Presets live in `maul_preset.ini` with sections:
- `[FM]`
- `[AM]`
- `[SSB]`

Each section contains the same set of controls (some have more effect in certain modes).

Example keys:
```ini
[FM]
hpHz=140
lpHz=3000
eqOn=1
eqLowDb=3
...
outGainDb=10
```

---

## DSP chain (what the app actually does)

The processing order is:

1. **Band-limit filters**
   - High-pass at `hpHz`
   - Low-pass at `lpHz` (2-pole)
2. **Gate / expander**
3. **3-band EQ**
4. **Pre-emphasis bite** (subtle differentiator to add “edge”)
5. **Compression stage 1**
6. **Compression stage 2**
7. **Exciter** (adds harmonics in a presence band)
8. **De-esser** (reduces sibilant energy)
9. **Mode character**
   - **AM**: asymmetric clip + saturation (more “AM station”)
   - **FM/SSB**: gentle saturation (less splatter)
10. **MonitorSim (optional)** — headphone “radio speaker” simulation
11. **Output gain**
12. **Lookahead limiter (final safety)** — enforces ceiling

> Important: The **limiter is last** so the ceiling actually holds, even after saturation/monitor shaping.

---

## What each control means (in detail)

### Output
#### `outGainDb`
Overall output level trim before the final limiter.  
Use this to hit your loudness target. If you hear crunch/flatness, back it off.

---

### Bandwidth
#### `hpHz`
High-pass cutoff in Hz (removes rumble/mud).  
- Higher = tighter/cleaner, less bass
- Lower = thicker but can get boomy

#### `lpHz`
Low-pass cutoff in Hz (removes hiss/fizz).  
- Lower = more “radio / comms” bandwidth
- Higher = more natural/bright

---

### EQ (3 bands)
#### `eqOn`
Enable/disable EQ.

#### `eqLowDb` (~220 Hz)
Adds/removes chest and thickness.  
Too high = muddy.

#### `eqNasalDb` (~800 Hz)
Controls “honk/box/nasal.”  
More negative usually sounds smoother.

#### `eqPresDb` (~2400 Hz)
Controls presence/clarity/intelligibility.  
Too high can become harsh.

---

### Gate / Expander
#### `gateOn`
Enable/disable gate.

#### `gateThreshDb`
Level where gating starts.  
- More negative = opens more easily (less gating)
- Less negative = gates more (cleaner but can chop words)

#### `gateRatio`
Strength of gating.  
- `1.0` ≈ off
- `3–6` typical
- `8–12` heavy

#### `gateFloorDb`
Maximum attenuation when gated (negative dB).  
- Example: `-30` means “up to 30 dB reduction”
- `0` means no reduction (gate can’t do anything)

---

### Compression (two stages)
Compression is what makes voice feel **big** (dense + consistent), not just loud.

#### `comp1On`, `c1Thresh`, `c1Ratio`, `c1Makeup`
Stage 1 = “body” compression (thickness + density).
- Lower threshold = compresses more often
- Ratio `2–4` = thick and natural
- Makeup restores loudness after gain reduction

#### `comp2On`, `c2Thresh`, `c2Ratio`, `c2Makeup`
Stage 2 = “peak grab / forwardness.”
- Higher ratio catches peaks harder
- Often used to keep voice consistently “in your face”

---

### De-esser
#### `deessOn`
Enable/disable.

#### `deessThreshDb`
When sibilance reduction starts.  
Lower = more active.

#### `deessAmt`
Strength (0..1).  
Too high can dull consonants.

---

### Exciter
Adds harmonic “bite” mainly in a presence band.

#### `excOn`
Enable/disable.

#### `excAmt`
How much harmonic content is added (0..1).

#### `excDrive`
How hard the band is driven into saturation.  
More drive = more edge; too much = harsh.

---

### Limiter (final safety)
#### `limOn`
Enable/disable.

#### `limCeilDb`
Final ceiling (negative dBFS).  
- FM often uses lower ceilings like `-3`
- AM/SSB commonly `-1` to `-1.5`

#### `limLookaheadMs`
Lookahead time. More lookahead catches peaks cleaner.

#### `limReleaseMs`
How quickly limiter recovers.  
Too fast = pumping; too slow = held-down sound.

---

### Saturation / AM clip controls
#### `satDrive`
Saturation intensity (all modes).  
Higher = thicker/dirtier; too high = flat.

#### `amPosCeil`, `amNegCeil` (AM mode)
Asymmetric clipping ceilings.  
Used only in AM mode to create the “big AM station” texture.

---

### Monitoring
#### `monSim`
A headphone “radio speaker” simulator (band-limits + honk peak).  
Use for listening/monitoring only.

> If you’re feeding a TX chain via VB-Cable, you usually want `monSim=0`.

---

## Recommended tuning workflow (“Big as” method)

1. **Set bandwidth first** (`hpHz`, `lpHz`)
2. **EQ next** (low thickness, nasal cut, presence)
3. **Dial compression** (C1 for body, C2 for density)
4. Add **exciter** if you need more “edge”
5. Add **de-esser** if S/T sounds are sharp
6. Adjust **outGainDb** for loudness
7. Set limiter ceiling to protect your signal

If it’s loud but not “big”:  
✅ lower **C1 threshold** slightly and reduce **outGainDb** a bit.

If it’s harsh / crunchy:  
✅ reduce **outGainDb**, then reduce **excAmt** or **satDrive**.

---

## Build instructions (Visual Studio)

### Requirements
- Windows 10/11
- Visual Studio 2019/2022
- C++20

### Linker dependencies
Add these to your project:
- `ole32.lib`
- `uuid.lib`
- `avrt.lib`

### Notes
- Uses WASAPI **shared mode**, so it works with normal Windows audio devices.
- Uses default devices (Console role first), prints device names on launch.

---

## Troubleshooting

### “No change when I tweak settings”
- Make sure you’re monitoring the app output **from the right place**:
  - If you routed the app to VB-Cable, listen to **CABLE Output**
  - If not, listen to your selected Windows output device
- If `monSim` is on, it can mask changes in `hpHz/lpHz` (it imposes its own band shape).

### “Audio is delayed / gets laggy”
- Use smaller buffer sizes if possible.
- Avoid huge lookahead values.
- Ensure your system isn’t overloaded.

### “AirSpy/SDR doesn’t hear it”
- Confirm:
  - `MAULSOUNDRADIO.exe` output is set to **CABLE Input**
  - SDR app input is **CABLE Output**

### “Gate does nothing”
- Check `gateFloorDb` is negative (e.g., `-30`).  
  `0` means it can’t attenuate.

### “Compressor does nothing”
- Your thresholds may be too high (close to 0 dBFS).  
  Typical voice thresholds are negative (e.g., `-30` to `-15`).

---

## Safety / disclaimer

This is a real-time audio tool. If you use it for transmitting, you are responsible for:
- Staying within legal bandwidth/spectral limits
- Avoiding over-deviation / splatter
- Keeping levels safe for your gear



