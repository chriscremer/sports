"""Persist ball exports beside the analysis, including explicit unknown states."""
import csv
from pathlib import Path

def write_ball_csv(data,directory):
    path=Path(directory)/f"rally-{data['clip']['start']:g}-ball.csv"
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time_seconds','pixel_x','pixel_y','model_score','x_m','y_m','height_m',
                         'vx_m_s','vy_m_s','vz_m_s','depth_status'])
        for row in data.get('ball_track',[]):
            observed=row.get('ball') or {};estimated=row.get('reconstruction') or {}
            writer.writerow([row['t'],*observed.get('pixel',['','']),observed.get('model_score',''),
                *estimated.get('xyz',['','','']),*estimated.get('velocity',['','','']),
                'estimated' if estimated else '2d_only' if observed else 'unknown'])
    temporary.replace(path)
    return path
