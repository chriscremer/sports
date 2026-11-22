"""Validate reviewed rally events and render frame-aligned timeline controls."""
from html import escape


def rally_timeline(review, clip, time_offset=0, *, heading='Rally timeline', section=None):
    events = review['events']
    if review['clip'] != clip['id'] or len(events) < 2:
        raise ValueError('Rally review must describe this clip and include its boundaries')
    if events[0]['kind'] != 'serve' or events[-1]['kind'] != 'end':
        raise ValueError('Rally events must begin with a serve and finish with an end')
    if any(event['kind'] != 'hit' for event in events[1:-1]):
        raise ValueError('Only paddle hits belong between the boundaries')

    fps, duration = clip['fps'], clip['duration']
    previous_frame = -1
    for event in events:
        frame = event['frame']
        first, last = event['frame_interval']
        if not (previous_frame < frame < clip['frame_count'] and 0 <= first <= frame <= last < clip['frame_count']):
            raise ValueError(f'Invalid frame or uncertainty interval for rally event {frame}')
        if event['source_frame'] != clip['first_frame'] + frame:
            raise ValueError(f'Wrong source frame for rally event {frame}')
        if abs(event['source_seconds'] - (clip['source_start'] + frame / fps)) > .001:
            raise ValueError(f'Wrong source time for rally event {frame}')
        if event['kind'] != 'end' and event['team'] not in ('near', 'far'):
            raise ValueError(f'Missing team for rally event {frame}')
        previous_frame = frame

    def position(frame):
        return 100 * frame / fps / duration

    def time_label(seconds):
        if not time_offset:
            return f'{seconds:.2f}s'
        time = time_offset + seconds
        return f'{int(time) // 60}:{time % 60:05.2f}'

    def marker(event, name, symbol):
        frame = event['frame']
        low, high = event['frame_interval']
        uncertain = high - low > 1
        text = f'{name} · {time_label(frame / fps)}' + (' · approximate' if uncertain else '')
        hint = escape(f"{text}. {event['evidence']}", quote=True)
        seek = time_offset + (frame + .15) / fps
        marker_class = 'rally-mark approximate' if uncertain else 'rally-mark'
        return (f'<button type="button" class="{marker_class}" style="--at:{position(frame):.5f}%" '
                f'data-rally-frame="{frame}" data-rally-label="{escape(text, quote=True)}" '
                f'data-review-seek="{seek:.6f}" aria-label="{escape(text, quote=True)}" title="{hint}">'
                f'<span aria-hidden="true">{symbol}</span></button>')

    start, end = events[0]['frame'], events[-1]['frame']
    bounds = marker(events[0], 'Rally start', '▶') + marker(events[-1], 'Rally end', '■')
    lanes = [f'<div class="rally-label">In play</div><div class="rally-lane bounds">'
             f'<span class="rally-span" style="--start:{position(start):.5f}%;--end:{position(end):.5f}%"></span>'
             f'{bounds}</div>']
    for team in ('near', 'far'):
        hits = [event for event in events if event['team'] == team and event['kind'] != 'end']
        marks = ''.join(marker(event, f'{team.title()} team {event["kind"]}', 'S' if event['kind'] == 'serve' else '●')
                        for event in hits)
        lanes.append(f'<div class="rally-label {team}">{team.title()} hits <small>{len(hits)}</small></div>'
                     f'<div class="rally-lane {team}">{marks}</div>')

    mid = duration / 2
    times = [time_label(t) for t in (0, mid, duration)] if time_offset else ['0s', f'{mid:.1f}s', f'{duration:.1f}s']
    ticks = ''.join(f'<span>{t}</span>' for t in times)
    return (
        f'<section class="rally-timeline" aria-label="Rally events" data-fps="{fps}" '
        f'data-duration="{duration}" data-offset="{time_offset}" data-start-frame="{start}" data-end-frame="{end}"'
        + (f' data-rally-section="{section}" hidden' if section is not None else '')
        + f'><div class="rally-heading"><strong>{escape(heading)}</strong>'
        f'<span>{time_label(start / fps)} start · {time_label(end / fps)} end</span></div>'
        f'<div class="rally-grid">{"".join(lanes)}<div></div><div class="rally-ticks">{ticks}</div></div>'
        '<div class="rally-footer"><span data-rally-phase>Before serve</span>'
        '<span>Click a marker to seek · outlined markers are approximate</span></div></section>\n'
    )
