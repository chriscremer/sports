export function sourceMatrixAt({shots}, time) {
  const shot = shots.find(({start, end}) => start <= time && time < end);
  if (!shot || shot.matrix) return shot?.matrix;
  const {samples} = shot;
  let lo = 0, hi = samples.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid][0] < time) lo = mid + 1;
    else hi = mid;
  }
  const before = samples[lo - 1], after = samples[lo];
  const left = before && time - before[0] <= .27 ? before : null;
  const right = after && after[0] - time <= .27 ? after : null;
  if (!left || !right) return (left || right)?.[1];
  const weight = (time - left[0]) / (right[0] - left[0]);
  return left[1].map((value, i) => value + weight * (right[1][i] - value));
}

export function projectSource(matrix, [x, y, z]) {
  const depth = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
  if (!(depth > 0)) return null;
  return [
    (matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]) / depth,
    (matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]) / depth,
  ];
}

export function createMatchOverlay(canvas, connections, colors) {
  const ctx = canvas.getContext('2d');
  let lastTime, painted = false;

  return (calibration, time, players, ball, visible, force) => {
    if (visible && !force && time === lastTime) return;
    lastTime = visible ? time : undefined;
    if (painted) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      painted = false;
    }
    const matrix = visible && calibration && sourceMatrixAt(calibration, time);
    if (!matrix || (!players.length && !ball)) return;
    ctx.setTransform(canvas.width / calibration.width, 0, 0, canvas.height / calibration.height, 0, 0);
    ctx.lineCap = ctx.lineJoin = 'round';
    const dot = (at, radius) => {
      ctx.moveTo(at[0] + radius, at[1]);
      ctx.arc(...at, radius, 0, 2 * Math.PI);
    };
    for (const player of players) {
      const points = player.joints3d.map(point => point && projectSource(matrix, point));
      const feet = projectSource(matrix, [...player.xy, 0]);
      ctx.globalAlpha = .85 * (player.opacity ?? (player.tracking_status === 'interpolated' ? .55 : 1));
      ctx.beginPath();
      for (const [start, end] of connections) {
        if (!points[start] || !points[end]) continue;
        ctx.moveTo(...points[start]);
        ctx.lineTo(...points[end]);
      }
      if (feet) dot(feet, 3.5);
      ctx.strokeStyle = '#101819';
      ctx.lineWidth = 4.2;
      ctx.stroke();
      ctx.strokeStyle = colors[player.team];
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    const center = ball && projectSource(matrix, ball.xyz);
    if (center) {
      ctx.globalAlpha = 1;
      ctx.beginPath();
      dot(center, 4);
      ctx.strokeStyle = '#101819';
      ctx.lineWidth = 4;
      ctx.stroke();
      // Fitted and predicted positions use the same center marker.
      if (!ball.predicted) {
        ctx.fillStyle = '#e3fa68';
        ctx.fill();
      } else {
        ctx.strokeStyle = '#e3fa68';
        ctx.lineWidth = 1.65;
        ctx.stroke();
        ctx.beginPath();
        dot(center, .7);
        ctx.fillStyle = '#e3fa68';
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    painted = true;
  };
}
