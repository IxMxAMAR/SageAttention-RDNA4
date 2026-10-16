# Testing on an RX 9060 XT (gfx1200)

You don't need to be a developer for this. It's install, set one environment variable, run one
script, and send back two files. Budget about 15 minutes.

**Why it needs you.** The hand-written kernels in this package were built and measured on an
RX 9070 XT (`gfx1201`). The RX 9060 XT is `gfx1200`: the same RDNA4 instruction set, a different
chip. Since v0.4.0 the wheel carries `gfx1200` builds of every kernel the default paths use, and
their machine code is byte-identical to the `gfx1201` builds. But nobody has run them on a 9060 XT
yet, so they ship switched off. Until someone confirms them, a 9060 XT keeps getting PR #368's
kernels, exactly as before.

If you'd rather not, that's completely fine: nothing changes for you.

## 1. Install

You need ComfyUI already working on Windows with PyTorch `2.13.0+rocm10.0.0` (see the main README).

Download the v0.4.0 wheel from [Releases](../../../releases) and install it into ComfyUI's own
Python. Close ComfyUI first. `--no-deps` matters, it keeps pip away from your PyTorch:

```powershell
C:\ComfyUI\venv\Scripts\python.exe -m pip install --no-deps --force-reinstall `
  sageattention-2.2.0+amd.gfx12.4-cp312-cp312-win_amd64.whl
```

Use your own ComfyUI path. On the portable build the Python is `python_embeded\python.exe`.

If you once set `HSA_OVERRIDE_GFX_VERSION` for an older Radeon, remove it; it is not meant for an
RDNA4 card:

```powershell
[Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION', $null, 'User')
Remove-Item Env:HSA_OVERRIDE_GFX_VERSION -ErrorAction SilentlyContinue
```

## 2. Turn the gfx1200 kernels on

The switch is read once, when Python imports the package, so set it before the probe or ComfyUI
starts:

```powershell
$env:SAGEATTN_SK1_GFX1200 = "1"
```

- **Unset, on a 9060 XT:** today's behaviour, PR #368's kernels.
- **`1`, on a 9060 XT:** the `gfx1200` builds.
- **`1`, on any other card:** no effect. The card always picks its own builds; the switch only
  permits `gfx1200`.

Going back is just removing the variable.

## 3. Run the probe

Get [`tools/gfx1200_probe.py`](../tools/gfx1200_probe.py) from this repository (the raw file is
enough), then run it with the same Python that runs ComfyUI, once with the switch on and once with
it off:

```powershell
$env:SAGEATTN_SK1_GFX1200 = "1"
C:\ComfyUI\venv\Scripts\python.exe gfx1200_probe.py --out gfx1200_report.json

Remove-Item Env:SAGEATTN_SK1_GFX1200
C:\ComfyUI\venv\Scripts\python.exe gfx1200_probe.py --out gfx1200_report_switch_off.json
```

Use `--out`, not `>`. In Windows PowerShell, `python ... > file.json` mixes warnings into the file
and breaks the JSON.

It takes a few seconds and uses random numbers only: no model, no workflow, no images. Paths in the
report have your home folder replaced with `~`.

## 4. What proves the kernels ran

The probe prints a summary like this one, taken on an RX 9070 XT, so the arch reads `gfx1201`:

```
device arch       gfx1201
selected arch     gfx1201 (None = the package will refuse)
switch            ON | SAGEATTN_SK1_GFX1200 = 1
gfx1200 objects   ALL PRESENT
would load        sk1_t4a1s.gfx1201.hsaco
------------------------------------------------------------------------------
served by         {'_BACKEND_I8': 'sk1_t6i.gfx1201.hsaco'}
output            finite | absmax 0.0927
vs fallback       max |diff| = 0.0057
  OK: the gfx1201 objects served this call
```

On a 9060 XT with the switch on, it should say:

| line | expected | what it means |
|---|---|---|
| `device arch` | `gfx1200` | the package sees your card correctly; if not, stop, nothing else matters |
| `switch` | `ON` | the opt-in was read |
| `gfx1200 objects` | `ALL PRESENT` | the gfx1200 builds are installed |
| `would load` | a name ending `_1200.hsaco` | the package picked the gfx1200 builds |
| **`served by`** | **names ending `_1200.hsaco`** | **the proof: these are the code objects that actually ran your call. Empty means it fell back** |
| `output` | `finite` | no NaN or Inf |
| `vs fallback` | a small number, around 0.01 | close to PR #368's result on the same input |

If `served by` is empty or shows `.gfx1201.hsaco` names on a 9060 XT, that is the finding. Don't try
to fix it, just send the files.

## 5. What to send back

Open an issue on this repository titled **"gfx1200 test report"** and attach:

1. `gfx1200_report.json` (switch on) and `gfx1200_report_switch_off.json` (switch off);
2. whether ComfyUI still renders normally with `SAGEATTN_SK1_GFX1200=1` set before you start it.
   One ordinary image is enough;
3. anything that looked wrong: a crash, a black image, or a log line mentioning
   `hipErrorNoBinaryForGpu`.

## 6. Optional, if you want to go further

With a checkout of this repository, the wheel installed and the switch on. Run the tests from a
folder *outside* the checkout: the checkout's own `sageattention\` folder has no compiled
extensions and would shadow the installed package.

```powershell
cd $HOME
C:\ComfyUI\venv\Scripts\python.exe -m pytest <path-to>\SageAttention-RDNA4\tests -q
C:\ComfyUI\venv\Scripts\python.exe <path-to>\SageAttention-RDNA4\bench\bench_attention.py
C:\ComfyUI\venv\Scripts\python.exe <path-to>\SageAttention-RDNA4\bench\bench_attention.py --preset h3
```

The tests check accuracy against PyTorch's attention and bit-identity across layouts. The
benchmark prints our time against PR #368's and PyTorch's, and a `kernel` column that says whether
our kernel served each row. Those numbers would be the first speed measurements on a 9060 XT. It
has fewer compute units than a 9070 XT, so the margins may differ.
