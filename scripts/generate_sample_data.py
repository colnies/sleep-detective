"""
Generate synthetic Fitbit sleep data and a matching habit log for trying out
Sleep Detective. Writes both CSVs to sample_data/ by default.

    python scripts/generate_sample_data.py [output_dir] [num_days]
"""

import csv
import os
import sys
import random
from datetime import datetime, timedelta


def generate_sleep_data(output_file: str, num_days: int = 365) -> list:
    """Generate synthetic Fitbit sleep data in CSV format."""
    
    end_date = datetime(2025, 12, 17)
    start_date = end_date - timedelta(days=num_days - 1)
    
    dates = []
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sleep_log_entry_id', 'timestamp', 'overall_score', 
            'composition_score', 'revitalization_score', 'duration_score',
            'deep_sleep_in_minutes', 'resting_heart_rate', 'restlessness'
        ])
        
        current_date = start_date
        sleep_log_id = 50000000000
        
        base_rhr = random.randint(58, 72)
        
        for day_num in range(num_days):
            date_str = current_date.strftime("%Y-%m-%d")
            dates.append(date_str)
            
            hour = random.choices([6, 7, 8, 9, 10], weights=[15, 30, 30, 15, 10])[0]
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            timestamp = f"{date_str}T{hour:02d}:{minute:02d}:{second:02d}Z"
            
            day_of_week = current_date.weekday()
            is_weekend = day_of_week >= 5
            
            if is_weekend:
                base_score = random.gauss(78, 8)
            else:
                base_score = random.gauss(75, 10)
            
            month = current_date.month
            if month in [12, 1, 2]:
                base_score -= random.uniform(0, 3)
            elif month in [6, 7, 8]:
                base_score += random.uniform(0, 2)
            
            sleep_score = max(40, min(95, base_score))
            
            if sleep_score >= 85:
                deep_sleep = random.randint(90, 150)
            elif sleep_score >= 70:
                deep_sleep = random.randint(60, 110)
            elif sleep_score >= 55:
                deep_sleep = random.randint(40, 80)
            else:
                deep_sleep = random.randint(20, 60)
            
            rhr_variation = random.gauss(0, 3)
            if not is_weekend:
                rhr_variation += random.uniform(0, 2)
            rhr = max(50, min(85, base_rhr + rhr_variation))
            
            if sleep_score >= 85:
                restlessness = random.uniform(0.04, 0.08)
            elif sleep_score >= 70:
                restlessness = random.uniform(0.06, 0.10)
            elif sleep_score >= 55:
                restlessness = random.uniform(0.08, 0.14)
            else:
                restlessness = random.uniform(0.12, 0.25)
            
            writer.writerow([
                sleep_log_id,
                timestamp,
                int(sleep_score),
                '',  # composition_score (often empty in real data)
                int(sleep_score),  # revitalization_score
                '',  # duration_score (often empty)
                deep_sleep,
                int(rhr),
                f"{restlessness:.17f}"
            ])
            
            sleep_log_id += random.randint(8000000, 12000000)
            current_date += timedelta(days=1)
            
            if day_num % 30 == 0:
                base_rhr += random.uniform(-1, 1)
                base_rhr = max(55, min(75, base_rhr))
    
    return dates


def generate_habit_data(dates: list, output_file: str):
    """Generate synthetic habit data for the given dates with correlations."""
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'caffeine_time', 'magnesium_time', 'exercise_done',
            'exercise_time', 'screen_time_before_bed', 'alcohol_drinks', 'stress_level'
        ])
        
        for date in dates:
            dt = datetime.strptime(date, "%Y-%m-%d")
            day_of_week = dt.weekday()
            is_weekend = day_of_week >= 5
            
            caffeine_time = ""
            if random.random() < 0.85:
                if is_weekend:
                    hour = random.choices(
                        [8, 9, 10, 11, 12, 13, 14, 15],
                        weights=[10, 20, 25, 15, 10, 8, 7, 5]
                    )[0]
                else:
                    hour = random.choices(
                        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                        weights=[5, 15, 20, 15, 10, 8, 7, 6, 5, 4, 3, 2]
                    )[0]
                minute = random.randint(0, 59)
                caffeine_time = f"{hour + minute/60:.2f}"
            
            magnesium_time = ""
            if random.random() < 0.45:
                hour = random.choices(
                    [17, 18, 19, 20, 21, 22],
                    weights=[5, 15, 25, 30, 20, 5]
                )[0]
                minute = random.randint(0, 59)
                magnesium_time = f"{hour + minute/60:.2f}"
            
            if is_weekend:
                exercise_prob = 0.45
            else:
                exercise_prob = 0.55
            
            exercise_done = random.random() < exercise_prob
            exercise_time = ""
            if exercise_done:
                hour = random.choices(
                    [6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19],
                    weights=[5, 10, 15, 15, 10, 5, 5, 10, 10, 10, 5]
                )[0]
                minute = random.randint(0, 59)
                exercise_time = f"{hour + minute/60:.2f}"
            
            if is_weekend:
                screen_time = random.choices(
                    [15, 30, 45, 60, 75, 90, 120],
                    weights=[5, 10, 15, 20, 20, 20, 10]
                )[0]
            else:
                screen_time = random.choices(
                    [10, 20, 30, 45, 60, 75, 90, 120],
                    weights=[5, 15, 25, 20, 15, 10, 7, 3]
                )[0]
            
            if is_weekend:
                alcohol = random.choices(
                    [0, 1, 2, 3, 4, 5],
                    weights=[40, 25, 20, 10, 4, 1]
                )[0]
            else:
                alcohol = random.choices(
                    [0, 1, 2, 3],
                    weights=[70, 20, 8, 2]
                )[0]
            
            if is_weekend:
                stress = random.choices(range(1, 8), weights=[5, 10, 20, 25, 20, 15, 5])[0]
            else:
                stress = random.choices(range(2, 10), weights=[5, 10, 15, 20, 20, 15, 10, 5])[0]
            
            writer.writerow([
                date,
                caffeine_time,
                magnesium_time,
                str(exercise_done).lower(),
                exercise_time,
                screen_time,
                alcohol,
                stress
            ])
    
    return len(dates)


def main():
    random.seed(42)
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    num_records = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    os.makedirs(output_dir, exist_ok=True)
    sleep_file = os.path.join(output_dir, "fitbit_sleep_data.csv")
    habit_file = os.path.join(output_dir, "daily_habit_log.csv")
    
    print("=" * 60)
    print("SLEEP DATA GENERATOR")
    print("=" * 60)
    print()
    print(f"  Generating {num_records:,} days of synthetic data...")
    print()
    
    print(f"  Creating {sleep_file}...")
    dates = generate_sleep_data(sleep_file, num_records)
    print(f"    -> Generated {len(dates):,} sleep records")
    print(f"    -> Date range: {dates[0]} to {dates[-1]}")
    print()
    
    print(f"  Creating {habit_file}...")
    count = generate_habit_data(dates, habit_file)
    print(f"    -> Generated {count:,} habit records")
    print()
    
    print("  Data includes:")
    print("    Sleep: score, deep_sleep_minutes, resting_heart_rate, restlessness")
    print("    Habits: caffeine, magnesium, exercise, screen_time, alcohol, stress")
    print()
    print("  Built-in patterns:")
    print("    - Weekend vs weekday variations")
    print("    - Seasonal variations (winter slightly worse)")
    print("    - Correlations between metrics (high score = low restlessness)")
    print()
    print("=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
