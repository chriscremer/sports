const lerp = (a, b, weight) => a + (b - a) * weight;
const lerpPoint = (a, b, weight) => a.map((value, i) => lerp(value, b[i], weight));

export function samplePlayers(frames, nearestIndex, time, sampleRate) {
  const nearest = frames[nearestIndex];
  if (!nearest) return [];

  const index = time < nearest.t ? nearestIndex - 1 : nearestIndex;
  const before = frames[index];
  const after = frames[index + 1];
  const gap = before && after ? after.t - before.t : 0;
  if (gap <= 0 || gap > 1.5 / sampleRate) return nearest.players;

  const weight = Math.max(0, Math.min(1, (time - before.t) / gap));
  const useBefore = nearest === before;
  const otherPlayers = useBefore ? after.players : before.players;
  return nearest.players.map(player => {
    const other = otherPlayers.find(candidate => candidate.id === player.id && candidate.team === player.team);
    if (!other) return player;

    const [from, to] = useBefore ? [player, other] : [other, player];
    const opacity = endpoint => endpoint.tracking_status === 'interpolated' ? .55 : 1;
    return {
      ...player,
      xy: lerpPoint(from.xy, to.xy, weight),
      joints3d: player.joints3d.map((joint, i) => {
        const start = from.joints3d[i], end = to.joints3d[i];
        return start && end ? lerpPoint(start, end, weight) : joint;
      }),
      opacity: lerp(opacity(from), opacity(to), weight),
    };
  });
}

export function shortFlightBridges(segments, contactTimes) {
  const bridges = [];
  for (let i = 1; i < segments.length; i++) {
    const before = segments[i - 1], after = segments[i];
    const gap = after.start - before.end;
    if (gap <= 0 || gap > .085 || contactTimes.some(time => before.end < time && time < after.start)) continue;
    if (before.camera_shot !== after.camera_shot) continue;
    if (!before.support_times.length || !after.support_times.length) continue;
    const endDistance = Math.abs(before.end - before.support_times.at(-1));
    const startDistance = Math.abs(after.support_times[0] - after.start);
    if (endDistance > (before.max_support_distance_s ?? .12) || startDistance > (after.max_support_distance_s ?? .12)) continue;

    const distance = Math.hypot(...before.p1.map((value, j) => value - after.p0[j]));
    if (distance > Math.min(.95, .12 + 27 * gap)) continue;
    bridges.push({start: before.end, end: after.start, from: before.p1, to: after.p0});
  }
  return bridges;
}

export function sampleBridge(bridges, time) {
  const bridge = bridges.find(({start, end}) => start < time && time < end);
  return bridge ? lerpPoint(bridge.from, bridge.to, (time - bridge.start) / (bridge.end - bridge.start)) : null;
}

export function sampleRallyBall(rallies, time, motionTime = time) {
  const rally = rallies.find(({start, end}) => start <= time && time <= end);
  if (!rally) return null;
  const {samples} = rally;
  const target = Math.max(rally.start, Math.min(rally.end, motionTime));
  let lo = 0, hi = samples.length - 1;
  while (lo < hi) {
    const middle = (lo + hi) >> 1;
    if (samples[middle][0] < target) lo = middle + 1;
    else hi = middle;
  }
  const index = Math.max(0, lo - 1);
  const a = samples[index], b = samples[lo];
  const weight = lo === index ? 0 : (target - a[0]) / (b[0] - a[0]);
  return {
    xyz: lerpPoint(a.slice(1, 4), b.slice(1, 4), weight),
    predicted: Boolean((weight < .5 ? a : b)[4]),
    rally,
    index,
  };
}

// Let the 3D view advance during a repeated media-clock tick. The lead is
// capped to a source frame, and paused/seeking playback uses exact media time.
export function createMotionClock() {
  let lastTime = null, lastRate = null, lastTick = 0, displayed = 0;
  return (mediaTime, now, rate, running, frameDuration) => {
    if (!running || lastTime === null || mediaTime < lastTime || rate !== lastRate || mediaTime - lastTime > .2) {
      lastTime = mediaTime;
      lastRate = rate;
      lastTick = now;
      displayed = mediaTime;
      return mediaTime;
    }
    if (mediaTime !== lastTime) {
      lastTime = mediaTime;
      lastTick = now;
    }
    const interpolated = mediaTime + Math.min(frameDuration, (now - lastTick) * rate / 1000);
    displayed = Math.max(displayed, interpolated);
    return displayed;
  };
}
