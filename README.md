# Sleep Detective

Find out which of your daily habits line up with better (or worse) sleep. Sleep Detective
reads your Fitbit sleep scores alongside a simple habit log you keep, then shows how things
like caffeine timing, exercise, magnesium, and screen time relate to how well you slept.

You get:

- A walk-through of each analysis right in the window
- A text report (`sleep_analysis_report.txt`)
- Six charts saved as PNG images

## Download and run

No installation or Python needed.

1. Go to the [latest release](https://github.com/colnies/sleep-detective/releases/latest) and
   download the zip for your computer:
   - **Windows:** `SleepDetective-…-windows-x64.zip`
   - **Mac (Apple Silicon: M1 or newer):** `SleepDetective-…-macos-arm64.zip`
2. Unzip it. You'll get a `SleepDetective` folder.
3. Open the folder and double-click **SleepDetective**.

The folder includes sample data, so you can press Enter at every prompt to try it out right away.
Your report and charts are saved in the same folder.

**First launch warnings.** The app isn't signed by a paid developer certificate, so your computer
may warn you the first time:

- **Windows** ("Windows protected your PC"): click **More info**, then **Run anyway**.
- **Mac** ("can't be opened because Apple cannot check it"): open **System Settings → Privacy &
  Security**, scroll down, and click **Open Anyway** next to SleepDetective. Then double-click it
  again.

## Using your own data

You need two CSV files. The easiest way to use them is to put them in the `SleepDetective` folder
with these exact names, replacing the sample files:

| File | What it is |
| --- | --- |
| `fitbit_sleep_data.csv` | Your Fitbit sleep scores |
| `daily_habit_log.csv` | Your daily habits |

You can also type a different path when the app asks, or drag a file into the window to paste
its path.

### 1. Your Fitbit sleep data

Export your Fitbit data (from the Fitbit app or [Google Takeout](https://takeout.google.com)). In the
export, find the file named **`sleep_score.csv`**, then copy it into the `SleepDetective` folder
and rename it to `fitbit_sleep_data.csv`.

The columns Sleep Detective uses are `timestamp`, `overall_score`, `deep_sleep_in_minutes`,
`resting_heart_rate`, and `restlessness`. Other columns are ignored.

### 2. Your habit log

Fill this in yourself, one row per day, in Excel, Google Sheets, or Numbers. Start from
`habit_log_template.csv` and save it as CSV named `daily_habit_log.csv`.

| Column | What to enter | Example |
| --- | --- | --- |
| `date` | The day, as YYYY-MM-DD | `2025-01-01` |
| `caffeine_time` | When you had your last caffeine, in 24-hour decimal hours. Leave blank if none. | `14.5` (2:30 PM) |
| `magnesium_time` | When you took magnesium, same format. Leave blank if none. | `21` (9:00 PM) |
| `exercise_done` | Did you exercise? | `true` or `false` |
| `exercise_time` | When you exercised. Leave blank if you didn't. | `17.25` (5:15 PM) |
| `screen_time_before_bed` | Minutes of screen time in the hour or so before bed | `30` |
| `alcohol_drinks` | Number of drinks | `0` |
| `stress_level` | 1 (calm) to 10 (very stressed) | `4` |

Only days that appear in **both** files are analyzed, so make sure the dates line up with
your Fitbit data.

## For developers

Install from source (Python 3.9+):

```bash
pipx install git+https://github.com/colnies/sleep-detective
sleep-detective
```

Or, from a clone: `pip install -e .`, then run `sleep-detective` or `python -m sleep_detective`.

Regenerate the sample data with `python scripts/generate_sample_data.py [output_dir] [num_days]`.

### Building the executables

```bash
pip install ".[build]"
python build_exe.py
```

This writes `dist/SleepDetective(.exe)` and a shareable `dist/SleepDetective-<version>-<platform>.zip`.
PyInstaller only builds for the system it runs on. To get a Mac build, run it on a Mac.

Releases are built automatically. Bump `__version__` in `sleep_detective/__init__.py`, then push a
matching tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The [Build executables](.github/workflows/release.yml) workflow builds the Windows and macOS zips
and attaches them to a GitHub Release for that tag.

---

Created by Colin Nies.
