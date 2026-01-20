import os

from sleep_detective_lib import (
    DEFAULT_SLEEP_DATA_FILE,
    DEFAULT_HABIT_LOG_FILE,
    DEFAULT_OUTPUT_FILE,
    HabitRecord,
    load_fitbit_csv,
    load_habit_csv,
    merge_records,
    analyze_caffeine_impact,
    analyze_magnesium_impact,
    analyze_exercise_impact,
    analyze_screen_time_impact,
    analyze_restlessness_impact,
    analyze_heart_rate_impact,
    analyze_sleep_only,
    calculate_correlation,
    predict_sleep_score,
    get_correlation_strength,
    generate_report,
    generate_all_charts
)


class Stack:
    
    def __init__(self):
        self._items = []
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self._items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self._items[-1]
        return None
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)


nav_stack = Stack()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_user(message: str = "Press Enter to continue, or 'b' to go back...", allow_back: bool = True):
    print()
    if allow_back:
        response = input(f">>> {message} ").strip().lower()
    else:
        input(f">>> {message}")
        response = ""
    print()
    return response


def print_banner():
    clear_screen()
    print()
    print("=" * 60)
    print("   ____  _                   ____                      ")
    print("  / ___|| | ___  ___ _ __   / ___|  ___ ___  _ __ ___  ")
    print("  \\___ \\| |/ _ \\/ _ \\ '_ \\  \\___ \\ / __/ _ \\| '__/ _ \\ ")
    print("   ___) | |  __/  __/ |_) |  ___) | (_| (_) | | |  __/ ")
    print("  |____/|_|\\___|\\___| .__/  |____/ \\___\\___/|_|  \\___| ")
    print("                    |_|                                ")
    print("              D E T E C T I V E                        ")
    print("=" * 60)
    print()
    print("  Analyze your Fitbit sleep data to discover patterns")
    print("  and optimize your sleep quality.")
    print()
    print("  Features:")
    print("    * Sleep score analysis with real Fitbit data")
    print("    * Resting heart rate impact")
    print("    * Restlessness analysis")
    print("    * Deep sleep tracking")
    print("    * Habit correlation analysis")
    print()


def get_file_inputs():
    print("=" * 60)
    print("DATA FILE SELECTION")
    print("=" * 60)
    print()
    
    print(f"Default sleep data file: {DEFAULT_SLEEP_DATA_FILE}")
    sleep_file = input("Enter path to Fitbit sleep data (CSV) [press Enter for default]: ").strip()
    if not sleep_file:
        sleep_file = DEFAULT_SLEEP_DATA_FILE
    
    while not os.path.exists(sleep_file):
        print(f"\n  [!] File not found: {sleep_file}")
        print("  Tip: Export your Fitbit sleep data as CSV first.\n")
        sleep_file = input("Enter path to Fitbit sleep data (CSV): ").strip()
        if not sleep_file:
            sleep_file = DEFAULT_SLEEP_DATA_FILE
    
    print(f"  [OK] Found: {sleep_file}\n")
    
    print(f"Default habit log file: {DEFAULT_HABIT_LOG_FILE}")
    habit_file = input("Enter path to daily habit log (CSV) [press Enter for default]: ").strip()
    if not habit_file:
        habit_file = DEFAULT_HABIT_LOG_FILE
    
    while not os.path.exists(habit_file):
        print(f"\n  [!] File not found: {habit_file}")
        print("  Tip: Run 'python generate_data.py' first to create sample habits.\n")
        habit_file = input("Enter path to daily habit log (CSV): ").strip()
        if not habit_file:
            habit_file = DEFAULT_HABIT_LOG_FILE
    
    print(f"  [OK] Found: {habit_file}\n")
    
    print(f"Default output report: {DEFAULT_OUTPUT_FILE}")
    output_file = input("Enter path for output report [press Enter for default]: ").strip()
    if not output_file:
        output_file = DEFAULT_OUTPUT_FILE
    
    print(f"  [OK] Output will be saved to: {output_file}\n")
    
    return sleep_file, habit_file, output_file


def load_and_validate_data(sleep_file: str, habit_file: str) -> list:
    print("=" * 60)
    print("LOADING DATA FILES")
    print("=" * 60)
    print()
    
    print(f"  Reading sleep data from: {sleep_file}")
    sleep_records = load_fitbit_csv(sleep_file)
    print(f"    -> Loaded {len(sleep_records):,} sleep records")
    
    print(f"  Reading habit data from: {habit_file}")
    habit_records = load_habit_csv(habit_file)
    print(f"    -> Loaded {len(habit_records):,} habit records")
    
    print("  Merging records by date...")
    snapshots = merge_records(sleep_records, habit_records)
    print(f"    -> Created {len(snapshots):,} matched daily snapshots")
    
    valid_count = sum(1 for s in snapshots if s.is_valid())
    invalid_count = len(snapshots) - valid_count
    
    print(f"\n  [OK] Valid records: {valid_count:,}")
    if invalid_count > 0:
        print(f"  [!] Invalid records (skipped): {invalid_count:,}")
    
    return snapshots


def display_summary_stats(snapshots: list):
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print()
    
    scores = [s.sleep.sleep_score for s in snapshots]
    deep_sleep = [s.sleep.deep_sleep_minutes for s in snapshots]
    rhr = [s.sleep.resting_heart_rate for s in snapshots if s.sleep.resting_heart_rate > 0]
    restlessness = [s.sleep.restlessness for s in snapshots]
    
    quality_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
    for s in snapshots:
        quality_counts[s.sleep.get_quality_label()] += 1
    
    print(f"  Total days analyzed: {len(snapshots):,}")
    print(f"  Date range: {snapshots[0].date} to {snapshots[-1].date}")
    print()
    print(f"  Average Sleep Score:    {sum(scores) / len(scores):.1f}")
    print(f"  Best Night:             {max(scores):.0f}")
    print(f"  Worst Night:            {min(scores):.0f}")
    print()
    print(f"  Avg Deep Sleep:         {sum(deep_sleep) / len(deep_sleep):.0f} minutes")
    if rhr:
        print(f"  Avg Resting Heart Rate: {sum(rhr) / len(rhr):.0f} bpm")
    print(f"  Avg Restlessness:       {sum(restlessness) / len(restlessness):.3f}")
    print()
    print("  Sleep Quality Distribution:")
    print("  " + "-" * 50)
    
    max_count = max(quality_counts.values())
    for quality, count in quality_counts.items():
        pct = (count / len(snapshots)) * 100
        bar_len = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"    {quality:10s} |{bar:<30s}| {pct:5.1f}% ({count:,})")


def display_caffeine_analysis(snapshots: list):
    print("=" * 60)
    print("CAFFEINE IMPACT ANALYSIS")
    print("=" * 60)
    print()
    
    results = analyze_caffeine_impact(snapshots)
    
    labels = {"none": "No caffeine", "early": "Early (<2pm)", 
              "moderate": "Moderate (2-5pm)", "late": "Late (>5pm)"}
    
    print("  Sleep Score by Caffeine Timing:")
    print("  " + "-" * 50)
    
    if not results:
        print("    No caffeine data available")
        return
    
    scores = {cat: results[cat]["avg_score"] for cat in results}
    min_score = min(scores.values()) - 5 if scores else 0
    max_score = max(scores.values()) if scores else 100
    
    for cat in ["none", "early", "moderate", "late"]:
        if cat in results:
            data = results[cat]
            score = data['avg_score']
            count = data['count']
            bar_len = int(((score - min_score) / (max_score - min_score + 1)) * 25)
            bar = "#" * bar_len
            print(f"    {labels[cat]:17s} |{bar:<25s}| {score:.1f} (n={count})")
    
    if "early" in results and "late" in results:
        diff = results["early"]["avg_score"] - results["late"]["avg_score"]
        print()
        if diff > 5:
            print("  +---------------------------------------------------+")
            print(f"  |  INSIGHT: Your sleep is {diff:.0f} points better when    |")
            print("  |  you cut off caffeine before 2pm!                 |")
            print("  +---------------------------------------------------+")
        elif diff > 0:
            print(f"  -> Early caffeine correlates with better sleep (+{diff:.1f} pts)")


def display_magnesium_analysis(snapshots: list):
    print("=" * 60)
    print("MAGNESIUM IMPACT ANALYSIS")
    print("=" * 60)
    print()
    
    results = analyze_magnesium_impact(snapshots)
    
    labels = {"none": "No magnesium", "early": "Early (<6pm)",
              "moderate": "Evening (6-9pm)", "late": "Late (>9pm)"}
    
    print("  Sleep Score by Magnesium Timing:")
    print("  " + "-" * 50)
    
    if not results:
        print("    No magnesium data available")
        return
    
    scores = {cat: results[cat]["avg_score"] for cat in results}
    min_score = min(scores.values()) - 5 if scores else 0
    max_score = max(scores.values()) if scores else 100
    
    for cat in ["none", "early", "moderate", "late"]:
        if cat in results:
            data = results[cat]
            score = data['avg_score']
            count = data['count']
            bar_len = int(((score - min_score) / (max_score - min_score + 1)) * 25)
            bar = "#" * bar_len
            print(f"    {labels[cat]:17s} |{bar:<25s}| {score:.1f} (n={count})")
    
    print()
    print("  Deep Sleep (minutes) by Magnesium Timing:")
    print("  " + "-" * 50)
    
    deep_scores = {cat: results[cat]["avg_deep_sleep"] for cat in results}
    min_deep = min(deep_scores.values()) - 5 if deep_scores else 0
    max_deep = max(deep_scores.values()) if deep_scores else 100
    
    for cat in ["none", "early", "moderate", "late"]:
        if cat in results:
            deep = results[cat]['avg_deep_sleep']
            bar_len = int(((deep - min_deep) / (max_deep - min_deep + 1)) * 25)
            bar = "#" * bar_len
            print(f"    {labels[cat]:17s} |{bar:<25s}| {deep:.0f} min")


def display_exercise_analysis(snapshots: list):
    print("=" * 60)
    print("EXERCISE IMPACT ANALYSIS")
    print("=" * 60)
    print()
    
    results = analyze_exercise_impact(snapshots)
    
    if "with_exercise" in results and "without_exercise" in results:
        with_ex = results["with_exercise"]
        without_ex = results["without_exercise"]
        diff = with_ex['avg_score'] - without_ex['avg_score']
        
        print("  Sleep Score Comparison:")
        print("  " + "-" * 50)
        
        min_score = min(with_ex['avg_score'], without_ex['avg_score']) - 5
        max_score = max(with_ex['avg_score'], without_ex['avg_score'])
        
        bar_len_without = int(((without_ex['avg_score'] - min_score) / (max_score - min_score + 1)) * 25)
        bar_len_with = int(((with_ex['avg_score'] - min_score) / (max_score - min_score + 1)) * 25)
        
        print(f"    Without exercise  |{'#' * bar_len_without:<25s}| {without_ex['avg_score']:.1f} (n={without_ex['count']})")
        print(f"    With exercise     |{'#' * bar_len_with:<25s}| {with_ex['avg_score']:.1f} (n={with_ex['count']})")
        print()
        print(f"    Difference: {diff:+.1f} points")
        
        if diff > 3:
            print()
            print("  +---------------------------------------------------+")
            print(f"  |  INSIGHT: Exercise improves your sleep by         |")
            print(f"  |  {diff:.0f} points on average!                           |")
            print("  +---------------------------------------------------+")
    else:
        print("  Not enough exercise data for comparison")


def display_restlessness_analysis(snapshots: list):
    print("=" * 60)
    print("RESTLESSNESS IMPACT ANALYSIS")
    print("=" * 60)
    print()
    
    results = analyze_restlessness_impact(snapshots)
    
    print("  Sleep Score by Restlessness Level:")
    print("  " + "-" * 50)
    
    if not results:
        print("    No restlessness data available")
        return
    
    scores = {cat: results[cat]["avg_score"] for cat in results}
    min_score = min(scores.values()) - 5 if scores else 0
    max_score = max(scores.values()) if scores else 100
    
    for cat in ["Low", "Moderate", "High"]:
        if cat in results:
            data = results[cat]
            score = data['avg_score']
            count = data['count']
            bar_len = int(((score - min_score) / (max_score - min_score + 1)) * 25)
            bar = "#" * bar_len
            print(f"    {cat:15s} |{bar:<25s}| {score:.1f} (n={count})")
    
    if "Low" in results and "High" in results:
        diff = results["Low"]["avg_score"] - results["High"]["avg_score"]
        print()
        if diff > 5:
            print("  +---------------------------------------------------+")
            print(f"  |  INSIGHT: Lower restlessness correlates with      |")
            print(f"  |  {diff:.0f} points better sleep!                         |")
            print("  +---------------------------------------------------+")


def display_heart_rate_analysis(snapshots: list):
    print("=" * 60)
    print("RESTING HEART RATE ANALYSIS")
    print("=" * 60)
    print()
    
    results = analyze_heart_rate_impact(snapshots)
    
    print("  Sleep Score by Resting Heart Rate:")
    print("  " + "-" * 50)
    
    if not results:
        print("    No heart rate data available")
        return
    
    scores = {cat: results[cat]["avg_score"] for cat in results}
    min_score = min(scores.values()) - 5 if scores else 0
    max_score = max(scores.values()) if scores else 100
    
    for cat in ["Athletic", "Good", "Average", "Elevated"]:
        if cat in results:
            data = results[cat]
            score = data['avg_score']
            count = data['count']
            bar_len = int(((score - min_score) / (max_score - min_score + 1)) * 25)
            bar = "#" * bar_len
            print(f"    {cat:15s} |{bar:<25s}| {score:.1f} (n={count})")
    
    print()
    print("  Heart Rate Categories:")
    print("    Athletic: <60 bpm | Good: 60-69 | Average: 70-79 | Elevated: 80+")


def display_screen_time_analysis(snapshots: list):
    print("=" * 60)
    print("SCREEN TIME IMPACT ANALYSIS")
    print("=" * 60)
    print()
    
    results = analyze_screen_time_impact(snapshots)
    
    print("  Score by Screen Time Level:")
    print("  " + "-" * 50)
    
    if not results:
        print("    No screen time data available")
        return
    
    labels = {"low": "Low (<30 min)", "moderate": "Moderate (30-60)", "high": "High (60+ min)"}
    
    scores = {cat: results[cat]["avg_score"] for cat in results}
    min_score = min(scores.values()) - 5 if scores else 0
    max_score = max(scores.values()) if scores else 100
    
    for cat in ["low", "moderate", "high"]:
        if cat in results:
            data = results[cat]
            score = data['avg_score']
            count = data['count']
            bar_len = int(((score - min_score) / (max_score - min_score + 1)) * 25)
            bar = "#" * bar_len
            print(f"    {labels[cat]:17s} |{bar:<25s}| {score:.1f} (n={count})")


def display_predictions(snapshots: list):
    print("=" * 60)
    print("SLEEP SCORE PREDICTIONS")
    print("=" * 60)
    print()
    
    scenarios = [
        ("OPTIMAL", {"caffeine_time": 8, "exercise_done": True, 
                     "magnesium_time": 20, "screen_time_before_bed": 15}),
        ("CHALLENGING", {"caffeine_time": 18, "exercise_done": False,
                         "magnesium_time": None, "screen_time_before_bed": 90}),
        ("MIXED", {"caffeine_time": 14, "exercise_done": True,
                   "magnesium_time": 22, "screen_time_before_bed": 45})
    ]
    
    predictions = []
    for name, habits in scenarios:
        pred = predict_sleep_score(habits, snapshots)
        predictions.append((name, pred))
    
    min_pred = min(p[1]["predicted_score"] for p in predictions) - 5
    max_pred = max(p[1]["predicted_score"] for p in predictions)
    
    print("  Predicted Scores for Different Scenarios:")
    print("  " + "-" * 50)
    
    for name, pred in predictions:
        score = pred["predicted_score"]
        bar_len = int(((score - min_pred) / (max_pred - min_pred + 1)) * 25)
        bar = "#" * bar_len
        print(f"    {name:12s} |{bar:<25s}| {score:.1f}")
    
    print()
    print(f"  Confidence: {predictions[0][1]['confidence']}")
    print()
    print("  Scenario Details:")
    print("    OPTIMAL: Early caffeine, exercise, evening magnesium, low screen")
    print("    CHALLENGING: Late caffeine, no exercise, no magnesium, high screen")
    print("    MIXED: Moderate habits across all factors")


def main():
    print_banner()
    wait_for_user("Press Enter to begin...", allow_back=False)
    
    clear_screen()
    print_banner()
    sleep_file, habit_file, output_file = get_file_inputs()
    wait_for_user("Press Enter to load data...", allow_back=False)
    
    clear_screen()
    snapshots = load_and_validate_data(sleep_file, habit_file)
    
    if not snapshots:
        print("\n  [ERROR] No valid data to analyze!")
        return 1
    
    wait_for_user("Press Enter to start analysis...", allow_back=False)
    
    screens = [
        ("Summary Statistics", lambda: display_summary_stats(snapshots)),
        ("Restlessness Impact", lambda: display_restlessness_analysis(snapshots)),
        ("Heart Rate Impact", lambda: display_heart_rate_analysis(snapshots)),
        ("Caffeine Impact", lambda: display_caffeine_analysis(snapshots)),
        ("Magnesium Impact", lambda: display_magnesium_analysis(snapshots)),
        ("Exercise Impact", lambda: display_exercise_analysis(snapshots)),
        ("Screen Time Impact", lambda: display_screen_time_analysis(snapshots)),
        ("Predictions", lambda: display_predictions(snapshots)),
    ]
    
    current = 0
    
    while current < len(screens):
        clear_screen()
        name, display_func = screens[current]
        display_func()
        
        nav_stack.push(current)
        
        if current < len(screens) - 1:
            next_name = screens[current + 1][0]
            response = wait_for_user(f"Enter to continue to {next_name}, or 'b' to go back...")
        else:
            response = wait_for_user("Enter to generate report, or 'b' to go back...")
        
        if response == 'b':
            nav_stack.pop()
            prev = nav_stack.pop()
            if prev is not None:
                current = prev
            else:
                current = 0
        else:
            current += 1
    
    clear_screen()
    print("=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)
    print()
    print(f"  Navigation history (stack): {nav_stack.size()} screens visited")
    print()
    
    print("  Generating analysis report...")
    generate_report(snapshots, output_file)
    print(f"  [OK] Report saved to: {output_file}")
    print()
    
    generate_all_charts(snapshots)
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print()
    print(f"  Total data points analyzed: {len(snapshots):,}")
    print(f"  Report saved to: {output_file}")
    print()
    print("  Generated charts:")
    print("    * chart_caffeine_impact.png")
    print("    * chart_magnesium_impact.png")
    print("    * chart_exercise_impact.png")
    print("    * chart_restlessness_impact.png")
    print("    * chart_sleep_timeline.png")
    print("    * chart_correlation_summary.png")
    print()
    print("  Thank you for using Sleep Score Detective!")
    print("=" * 60)
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
