import csv
from datetime import datetime
from collections import defaultdict
import statistics
import math

SLEEP_QUALITY_EXCELLENT = 85
SLEEP_QUALITY_GOOD = 70
SLEEP_QUALITY_FAIR = 55
SLEEP_QUALITY_POOR = 0

EARLY_CAFFEINE_CUTOFF = 14
LATE_CAFFEINE_CUTOFF = 17
EARLY_MAGNESIUM_CUTOFF = 18
LATE_MAGNESIUM_CUTOFF = 21

STRONG_CORRELATION = 0.5
MODERATE_CORRELATION = 0.3
WEAK_CORRELATION = 0.1

DEFAULT_SLEEP_DATA_FILE = "fitbit_sleep_data.csv"
DEFAULT_HABIT_LOG_FILE = "daily_habit_log.csv"
DEFAULT_OUTPUT_FILE = "sleep_analysis_report.txt"


class DataRecord:
    def __init__(self, date: str):
        self.date = date
        self._validated = False
    
    def validate(self) -> bool:
        try:
            datetime.strptime(self.date, "%Y-%m-%d")
            self._validated = True
            return True
        except ValueError:
            self._validated = False
            return False
    
    def is_valid(self) -> bool:
        return self._validated
    
    def to_dict(self) -> dict:
        return {"date": self.date}
    
    def __repr__(self):
        return f"{self.__class__.__name__}(date={self.date})"


class SleepRecord(DataRecord):
    def __init__(self, date: str, sleep_score: float,
                 deep_sleep_minutes: int = 0,
                 resting_heart_rate: int = 0,
                 restlessness: float = 0.0,
                 timestamp: str = None):
        super().__init__(date)
        self.sleep_score = sleep_score
        self.deep_sleep_minutes = deep_sleep_minutes
        self.resting_heart_rate = resting_heart_rate
        self.restlessness = restlessness
        self.timestamp = timestamp
    
    def validate(self) -> bool:
        if not super().validate():
            return False
        if not (0 <= self.sleep_score <= 100):
            self._validated = False
            return False
        if self.deep_sleep_minutes < 0:
            self._validated = False
            return False
        self._validated = True
        return True
    
    def get_quality_label(self) -> str:
        if self.sleep_score >= SLEEP_QUALITY_EXCELLENT:
            return "Excellent"
        elif self.sleep_score >= SLEEP_QUALITY_GOOD:
            return "Good"
        elif self.sleep_score >= SLEEP_QUALITY_FAIR:
            return "Fair"
        return "Poor"
    
    def get_restlessness_label(self) -> str:
        if self.restlessness < 0.08:
            return "Low"
        elif self.restlessness < 0.12:
            return "Moderate"
        return "High"
    
    def get_rhr_label(self) -> str:
        if self.resting_heart_rate == 0:
            return "Unknown"
        elif self.resting_heart_rate < 60:
            return "Athletic"
        elif self.resting_heart_rate < 70:
            return "Good"
        elif self.resting_heart_rate < 80:
            return "Average"
        return "Elevated"
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "sleep_score": self.sleep_score,
            "deep_sleep_minutes": self.deep_sleep_minutes,
            "resting_heart_rate": self.resting_heart_rate,
            "restlessness": self.restlessness,
            "quality_label": self.get_quality_label(),
            "restlessness_label": self.get_restlessness_label(),
            "rhr_label": self.get_rhr_label()
        })
        return base


class HabitRecord(DataRecord):
    def __init__(self, date: str, caffeine_time: float = None,
                 magnesium_time: float = None, exercise_done: bool = False,
                 exercise_time: float = None, screen_time_before_bed: int = 0,
                 alcohol_drinks: int = 0, stress_level: int = 5):
        super().__init__(date)
        self.caffeine_time = caffeine_time
        self.magnesium_time = magnesium_time
        self.exercise_done = exercise_done
        self.exercise_time = exercise_time
        self.screen_time_before_bed = screen_time_before_bed
        self.alcohol_drinks = alcohol_drinks
        self.stress_level = stress_level
    
    def validate(self) -> bool:
        if not super().validate():
            return False
        if self.caffeine_time is not None and not (0 <= self.caffeine_time <= 24):
            self._validated = False
            return False
        if self.magnesium_time is not None and not (0 <= self.magnesium_time <= 24):
            self._validated = False
            return False
        if not (1 <= self.stress_level <= 10):
            self._validated = False
            return False
        self._validated = True
        return True
    
    def caffeine_category(self) -> str:
        if self.caffeine_time is None:
            return "none"
        elif self.caffeine_time <= EARLY_CAFFEINE_CUTOFF:
            return "early"
        elif self.caffeine_time <= LATE_CAFFEINE_CUTOFF:
            return "moderate"
        return "late"
    
    def magnesium_category(self) -> str:
        if self.magnesium_time is None:
            return "none"
        elif self.magnesium_time <= EARLY_MAGNESIUM_CUTOFF:
            return "early"
        elif self.magnesium_time <= LATE_MAGNESIUM_CUTOFF:
            return "moderate"
        return "late"
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "caffeine_time": self.caffeine_time, "magnesium_time": self.magnesium_time,
            "exercise_done": self.exercise_done, "exercise_time": self.exercise_time,
            "screen_time_before_bed": self.screen_time_before_bed,
            "alcohol_drinks": self.alcohol_drinks, "stress_level": self.stress_level,
            "caffeine_category": self.caffeine_category(),
            "magnesium_category": self.magnesium_category()
        })
        return base


class DailySnapshot(DataRecord):
    def __init__(self, sleep_record: SleepRecord, habit_record: HabitRecord):
        if sleep_record.date != habit_record.date:
            raise ValueError("Sleep and habit records must have matching dates")
        super().__init__(sleep_record.date)
        self.sleep = sleep_record
        self.habits = habit_record
    
    def validate(self) -> bool:
        self._validated = self.sleep.validate() and self.habits.validate()
        return self._validated
    
    def to_dict(self) -> dict:
        return {"date": self.date, "sleep": self.sleep.to_dict(), "habits": self.habits.to_dict()}


def load_fitbit_csv(filepath: str) -> list:
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = row.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date = dt.strftime("%Y-%m-%d")
                else:
                    continue
                
                sleep_score = float(row.get('overall_score', 0))
                deep_sleep_minutes = int(float(row.get('deep_sleep_in_minutes', 0)))
                resting_heart_rate = int(float(row.get('resting_heart_rate', 0)))
                restlessness = float(row.get('restlessness', 0))
                
                record = SleepRecord(
                    date=date,
                    sleep_score=sleep_score,
                    deep_sleep_minutes=deep_sleep_minutes,
                    resting_heart_rate=resting_heart_rate,
                    restlessness=restlessness,
                    timestamp=timestamp
                )
                
                if record.validate():
                    records.append(record)
            except (ValueError, KeyError):
                continue
    
    return records


def load_habit_csv(filepath: str) -> list:
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                caffeine_time = float(row['caffeine_time']) if row.get('caffeine_time') and row['caffeine_time'] != '' else None
                magnesium_time = float(row['magnesium_time']) if row.get('magnesium_time') and row['magnesium_time'] != '' else None
                exercise_time = float(row['exercise_time']) if row.get('exercise_time') and row['exercise_time'] != '' else None
                
                record = HabitRecord(
                    date=row['date'],
                    caffeine_time=caffeine_time,
                    magnesium_time=magnesium_time,
                    exercise_done=row.get('exercise_done', '').lower() in ('true', '1', 'yes'),
                    exercise_time=exercise_time,
                    screen_time_before_bed=int(row.get('screen_time_before_bed', 0)),
                    alcohol_drinks=int(row.get('alcohol_drinks', 0)),
                    stress_level=int(row.get('stress_level', 5))
                )
                
                if record.validate():
                    records.append(record)
            except (ValueError, KeyError):
                continue
    
    return records


def merge_records(sleep_records: list, habit_records: list) -> list:
    sleep_by_date = {r.date: r for r in sleep_records}
    habit_by_date = {r.date: r for r in habit_records}
    common_dates = set(sleep_by_date.keys()) & set(habit_by_date.keys())
    
    snapshots = []
    for date in sorted(common_dates):
        snapshot = DailySnapshot(sleep_by_date[date], habit_by_date[date])
        if snapshot.validate():
            snapshots.append(snapshot)
    
    return snapshots


def analyze_caffeine_impact(snapshots: list) -> dict:
    categories = defaultdict(list)
    for snap in snapshots:
        cat = snap.habits.caffeine_category()
        categories[cat].append(snap.sleep.sleep_score)
    
    results = {}
    for cat, scores in categories.items():
        if scores:
            results[cat] = {
                "count": len(scores),
                "avg_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores),
                "max_score": max(scores)
            }
    return results


def analyze_magnesium_impact(snapshots: list) -> dict:
    categories = defaultdict(lambda: {"scores": [], "deep_sleep": []})
    for snap in snapshots:
        cat = snap.habits.magnesium_category()
        categories[cat]["scores"].append(snap.sleep.sleep_score)
        categories[cat]["deep_sleep"].append(snap.sleep.deep_sleep_minutes)
    
    results = {}
    for cat, data in categories.items():
        if data["scores"]:
            results[cat] = {
                "count": len(data["scores"]),
                "avg_score": statistics.mean(data["scores"]),
                "avg_deep_sleep": statistics.mean(data["deep_sleep"]),
                "std_dev": statistics.stdev(data["scores"]) if len(data["scores"]) > 1 else 0
            }
    return results


def analyze_exercise_impact(snapshots: list) -> dict:
    with_exercise = []
    without_exercise = []
    
    for snap in snapshots:
        if snap.habits.exercise_done:
            with_exercise.append(snap.sleep.sleep_score)
        else:
            without_exercise.append(snap.sleep.sleep_score)
    
    results = {}
    if with_exercise:
        results["with_exercise"] = {
            "count": len(with_exercise),
            "avg_score": statistics.mean(with_exercise),
            "std_dev": statistics.stdev(with_exercise) if len(with_exercise) > 1 else 0
        }
    if without_exercise:
        results["without_exercise"] = {
            "count": len(without_exercise),
            "avg_score": statistics.mean(without_exercise),
            "std_dev": statistics.stdev(without_exercise) if len(without_exercise) > 1 else 0
        }
    return results


def analyze_screen_time_impact(snapshots: list) -> dict:
    categories = defaultdict(list)
    for snap in snapshots:
        st = snap.habits.screen_time_before_bed
        if st < 30:
            categories["low"].append(snap.sleep.sleep_score)
        elif st < 60:
            categories["moderate"].append(snap.sleep.sleep_score)
        else:
            categories["high"].append(snap.sleep.sleep_score)
    
    results = {}
    for cat, scores in categories.items():
        if scores:
            results[cat] = {
                "count": len(scores),
                "avg_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
            }
    return results


def analyze_restlessness_impact(snapshots: list) -> dict:
    categories = defaultdict(list)
    for snap in snapshots:
        label = snap.sleep.get_restlessness_label()
        categories[label].append(snap.sleep.sleep_score)
    
    results = {}
    for cat, scores in categories.items():
        if scores:
            results[cat] = {
                "count": len(scores),
                "avg_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
            }
    return results


def analyze_heart_rate_impact(snapshots: list) -> dict:
    categories = defaultdict(list)
    for snap in snapshots:
        label = snap.sleep.get_rhr_label()
        if label != "Unknown":
            categories[label].append(snap.sleep.sleep_score)
    
    results = {}
    for cat, scores in categories.items():
        if scores:
            results[cat] = {
                "count": len(scores),
                "avg_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
            }
    return results


def analyze_sleep_only(sleep_records: list) -> dict:
    if not sleep_records:
        return {}
    
    scores = [r.sleep_score for r in sleep_records]
    deep_sleep = [r.deep_sleep_minutes for r in sleep_records]
    rhr = [r.resting_heart_rate for r in sleep_records if r.resting_heart_rate > 0]
    restlessness = [r.restlessness for r in sleep_records]
    
    quality_dist = defaultdict(int)
    for r in sleep_records:
        quality_dist[r.get_quality_label()] += 1
    
    results = {
        "total_records": len(sleep_records),
        "avg_score": statistics.mean(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
        "min_score": min(scores),
        "max_score": max(scores),
        "avg_deep_sleep_minutes": statistics.mean(deep_sleep),
        "avg_resting_heart_rate": statistics.mean(rhr) if rhr else 0,
        "avg_restlessness": statistics.mean(restlessness),
        "quality_distribution": dict(quality_dist)
    }
    
    return results


def calculate_correlation(x_values: list, y_values: list) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def get_correlation_strength(correlation: float) -> str:
    abs_corr = abs(correlation)
    if abs_corr >= STRONG_CORRELATION:
        strength = "Strong"
    elif abs_corr >= MODERATE_CORRELATION:
        strength = "Moderate"
    elif abs_corr >= WEAK_CORRELATION:
        strength = "Weak"
    else:
        strength = "Negligible"
    
    direction = "positive" if correlation > 0 else "negative"
    return f"{strength} {direction}"


def predict_sleep_score(habits: dict, historical_data: list) -> dict:
    base_score = 75.0
    adjustments = []
    
    if historical_data:
        base_score = statistics.mean([s.sleep.sleep_score for s in historical_data])
    
    caffeine = habits.get("caffeine_time")
    if caffeine is not None:
        if caffeine <= EARLY_CAFFEINE_CUTOFF:
            adjustments.append(("Early caffeine", 2))
        elif caffeine <= LATE_CAFFEINE_CUTOFF:
            adjustments.append(("Afternoon caffeine", -3))
        else:
            adjustments.append(("Late caffeine", -7))
    
    if habits.get("exercise_done"):
        adjustments.append(("Exercise completed", 3))
    
    magnesium = habits.get("magnesium_time")
    if magnesium is not None:
        if EARLY_MAGNESIUM_CUTOFF <= magnesium <= LATE_MAGNESIUM_CUTOFF:
            adjustments.append(("Evening magnesium", 2))
    
    screen_time = habits.get("screen_time_before_bed", 0)
    if screen_time < 30:
        adjustments.append(("Low screen time", 2))
    elif screen_time > 60:
        adjustments.append(("High screen time", -4))
    
    total_adjustment = sum(adj[1] for adj in adjustments)
    predicted = max(min(base_score + total_adjustment, 100), 0)
    
    return {
        "base_score": base_score,
        "adjustments": adjustments,
        "total_adjustment": total_adjustment,
        "predicted_score": predicted,
        "confidence": "Medium" if len(historical_data) > 30 else "Low"
    }


def generate_report(snapshots: list, output_file: str):
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SLEEP SCORE DETECTIVE - ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records Analyzed: {len(snapshots)}\n")
        
        if snapshots:
            dates = [s.date for s in snapshots]
            f.write(f"Date Range: {min(dates)} to {max(dates)}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 60 + "\n")
        
        if snapshots:
            scores = [s.sleep.sleep_score for s in snapshots]
            f.write(f"Average Sleep Score: {statistics.mean(scores):.1f}\n")
            f.write(f"Standard Deviation: {statistics.stdev(scores):.1f}\n")
            f.write(f"Minimum Score: {min(scores):.0f}\n")
            f.write(f"Maximum Score: {max(scores):.0f}\n\n")
            
            deep_sleep = [s.sleep.deep_sleep_minutes for s in snapshots]
            f.write(f"Average Deep Sleep: {statistics.mean(deep_sleep):.0f} minutes\n")
            
            rhr = [s.sleep.resting_heart_rate for s in snapshots if s.sleep.resting_heart_rate > 0]
            if rhr:
                f.write(f"Average Resting HR: {statistics.mean(rhr):.0f} bpm\n")
            
            restlessness = [s.sleep.restlessness for s in snapshots]
            f.write(f"Average Restlessness: {statistics.mean(restlessness):.3f}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("CAFFEINE IMPACT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        caffeine_results = analyze_caffeine_impact(snapshots)
        for cat, data in sorted(caffeine_results.items()):
            f.write(f"{cat.capitalize():15} - Avg: {data['avg_score']:.1f}, Count: {data['count']}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("MAGNESIUM IMPACT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        mag_results = analyze_magnesium_impact(snapshots)
        for cat, data in sorted(mag_results.items()):
            f.write(f"{cat.capitalize():15} - Avg Score: {data['avg_score']:.1f}, Avg Deep Sleep: {data['avg_deep_sleep']:.0f} min\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("EXERCISE IMPACT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        exercise_results = analyze_exercise_impact(snapshots)
        for cat, data in exercise_results.items():
            label = "With Exercise" if cat == "with_exercise" else "Without Exercise"
            f.write(f"{label:20} - Avg: {data['avg_score']:.1f}, Count: {data['count']}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("RESTLESSNESS IMPACT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        restless_results = analyze_restlessness_impact(snapshots)
        for cat in ["Low", "Moderate", "High"]:
            if cat in restless_results:
                data = restless_results[cat]
                f.write(f"{cat:15} - Avg Score: {data['avg_score']:.1f}, Count: {data['count']}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("HEART RATE IMPACT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        hr_results = analyze_heart_rate_impact(snapshots)
        for cat in ["Athletic", "Good", "Average", "Elevated"]:
            if cat in hr_results:
                data = hr_results[cat]
                f.write(f"{cat:15} - Avg Score: {data['avg_score']:.1f}, Count: {data['count']}\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")


def generate_all_charts(snapshots: list):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available - skipping chart generation")
        return
    
    print("  Generating caffeine impact chart...")
    generate_caffeine_chart(snapshots, plt)
    
    print("  Generating magnesium impact chart...")
    generate_magnesium_chart(snapshots, plt)
    
    print("  Generating exercise impact chart...")
    generate_exercise_chart(snapshots, plt)
    
    print("  Generating restlessness chart...")
    generate_restlessness_chart(snapshots, plt)
    
    print("  Generating sleep timeline chart...")
    generate_timeline_chart(snapshots, plt)
    
    print("  Generating correlation summary chart...")
    generate_correlation_chart(snapshots, plt)


def generate_caffeine_chart(snapshots: list, plt):
    results = analyze_caffeine_impact(snapshots)
    if not results:
        return
    
    categories = ["none", "early", "moderate", "late"]
    labels = ["No Caffeine", "Early (<2pm)", "Moderate (2-5pm)", "Late (>5pm)"]
    scores = []
    counts = []
    
    for cat in categories:
        if cat in results:
            scores.append(results[cat]["avg_score"])
            counts.append(results[cat]["count"])
        else:
            scores.append(0)
            counts.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, scores, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Average Sleep Score')
    ax.set_title('Impact of Caffeine Timing on Sleep Score')
    ax.set_ylim(0, 100)
    ax.axhline(y=statistics.mean([s.sleep.sleep_score for s in snapshots]), 
               color='gray', linestyle='--', label='Overall Average')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('chart_caffeine_impact.png', dpi=150)
    plt.close()


def generate_magnesium_chart(snapshots: list, plt):
    results = analyze_magnesium_impact(snapshots)
    if not results:
        return
    
    categories = ["none", "early", "moderate", "late"]
    labels = ["No Magnesium", "Early (<6pm)", "Evening (6-9pm)", "Late (>9pm)"]
    
    scores = []
    deep_sleep = []
    
    for cat in categories:
        if cat in results:
            scores.append(results[cat]["avg_score"])
            deep_sleep.append(results[cat]["avg_deep_sleep"])
        else:
            scores.append(0)
            deep_sleep.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bars1 = ax1.bar(labels, scores, color=['#95a5a6', '#3498db', '#2ecc71', '#9b59b6'])
    ax1.set_ylabel('Average Sleep Score')
    ax1.set_title('Sleep Score by Magnesium Timing')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=15)
    
    bars2 = ax2.bar(labels, deep_sleep, color=['#95a5a6', '#3498db', '#2ecc71', '#9b59b6'])
    ax2.set_ylabel('Average Deep Sleep (minutes)')
    ax2.set_title('Deep Sleep by Magnesium Timing')
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('chart_magnesium_impact.png', dpi=150)
    plt.close()


def generate_exercise_chart(snapshots: list, plt):
    results = analyze_exercise_impact(snapshots)
    if not results:
        return
    
    labels = []
    scores = []
    counts = []
    colors = []
    
    if "with_exercise" in results:
        labels.append("With Exercise")
        scores.append(results["with_exercise"]["avg_score"])
        counts.append(results["with_exercise"]["count"])
        colors.append('#2ecc71')
    
    if "without_exercise" in results:
        labels.append("Without Exercise")
        scores.append(results["without_exercise"]["avg_score"])
        counts.append(results["without_exercise"]["count"])
        colors.append('#e74c3c')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, scores, color=colors)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Average Sleep Score')
    ax.set_title('Impact of Exercise on Sleep Score')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('chart_exercise_impact.png', dpi=150)
    plt.close()


def generate_restlessness_chart(snapshots: list, plt):
    results = analyze_restlessness_impact(snapshots)
    if not results:
        return
    
    categories = ["Low", "Moderate", "High"]
    scores = []
    counts = []
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for cat in categories:
        if cat in results:
            scores.append(results[cat]["avg_score"])
            counts.append(results[cat]["count"])
        else:
            scores.append(0)
            counts.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, scores, color=colors)
    
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'n={count}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Average Sleep Score')
    ax.set_xlabel('Restlessness Level')
    ax.set_title('Impact of Restlessness on Sleep Score')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('chart_restlessness_impact.png', dpi=150)
    plt.close()

def generate_timeline_chart(snapshots: list, plt):
    if not snapshots:
        return
    
    sorted_snaps = sorted(snapshots, key=lambda s: s.date)
    dates = [datetime.strptime(s.date, "%Y-%m-%d") for s in sorted_snaps]
    scores = [s.sleep.sleep_score for s in sorted_snaps]
    
    if len(scores) >= 7:
        window = 7
        moving_avg = []
        for i in range(len(scores)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(statistics.mean(scores[start_idx:i+1]))
    else:
        moving_avg = scores
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.scatter(dates, scores, alpha=0.3, s=20, color='#3498db', label='Daily Score')
    ax.plot(dates, moving_avg, color='#e74c3c', linewidth=2, label='7-Day Moving Average')
    
    ax.axhline(y=SLEEP_QUALITY_EXCELLENT, color='green', linestyle=':', alpha=0.5, label='Excellent (85+)')
    ax.axhline(y=SLEEP_QUALITY_GOOD, color='orange', linestyle=':', alpha=0.5, label='Good (70+)')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sleep Score')
    ax.set_title('Sleep Score Over Time')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('chart_sleep_timeline.png', dpi=150)
    plt.close()


def generate_correlation_chart(snapshots: list, plt):
    if not snapshots:
        return
    
    scores = [s.sleep.sleep_score for s in snapshots]
    deep_sleep = [s.sleep.deep_sleep_minutes for s in snapshots]
    restlessness = [s.sleep.restlessness for s in snapshots]
    rhr = [s.sleep.resting_heart_rate for s in snapshots]
    screen_time = [s.habits.screen_time_before_bed for s in snapshots]
    stress = [s.habits.stress_level for s in snapshots]
    
    correlations = {
        'Deep Sleep': calculate_correlation(deep_sleep, scores),
        'Restlessness': calculate_correlation(restlessness, scores),
        'Resting HR': calculate_correlation(rhr, scores),
        'Screen Time': calculate_correlation(screen_time, scores),
        'Stress Level': calculate_correlation(stress, scores)
    }
    
    labels = list(correlations.keys())
    values = list(correlations.values())
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=colors)
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=MODERATE_CORRELATION, color='green', linestyle='--', alpha=0.3)
    ax.axvline(x=-MODERATE_CORRELATION, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Correlation with Sleep Score')
    ax.set_title('Factor Correlations with Sleep Score')
    ax.set_xlim(-1, 1)
    
    for bar, val in zip(bars, values):
        x_pos = val + 0.05 if val >= 0 else val - 0.05
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                ha=ha, va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('chart_correlation_summary.png', dpi=150)
    plt.close()
