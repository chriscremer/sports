// Both players use seconds from the start of the original recording.
export function matchPlayback(nativeVideo, source, duration, onChange) {
  const status = document.getElementById('media-status');
  const youtubeStage = document.getElementById('youtube-stage');
  const localStage = document.getElementById('local-stage');
  const fallback = document.getElementById('use-local');
  let player = null, ready = false, local = false, requestedTime = 0;
  let pendingUntil = 0;

  function switchToLocal(message) {
    if (local) return;
    requestedTime = clock.currentTime;
    local = true;
    if (ready) player.pauseVideo();
    youtubeStage.hidden = true;
    localStage.hidden = false;
    fallback.hidden = true;
    status.textContent = message;
    nativeVideo.addEventListener('loadedmetadata', () => {
      nativeVideo.currentTime = requestedTime;
      onChange();
    }, {once: true});
    nativeVideo.src = source.local_video;
    nativeVideo.load();
    onChange();
  }

  const clock = {
    get duration() { return duration; },
    get currentTime() {
      if (local && nativeVideo.readyState >= 1) return Math.min(duration, nativeVideo.currentTime);
      if (ready && !local && performance.now() >= pendingUntil) {
        const state = player.getPlayerState();
        if (state !== -1 && state !== 5) return Math.min(duration, player.getCurrentTime() || 0);
      }
      return requestedTime;
    },
    set currentTime(value) {
      requestedTime = Math.max(0, Math.min(duration, value));
      if (local && nativeVideo.readyState >= 1) nativeVideo.currentTime = requestedTime;
      if (!local && ready) {
        pendingUntil = performance.now() + 650;
        const state = player.getPlayerState();
        if (state === -1 || state === 5) player.cueVideoById({videoId: source.id, startSeconds: requestedTime, endSeconds: duration});
        else player.seekTo(requestedTime, true);
      }
    },
    get paused() { return local ? nativeVideo.paused : !ready || player.getPlayerState() !== 1; },
    get seeking() { return local ? nativeVideo.seeking : performance.now() < pendingUntil; },
    get readyState() { return local ? nativeVideo.readyState : ready ? 4 : 0; },
    get hasFrame() {
      if (local) return nativeVideo.readyState >= 2 && !nativeVideo.seeking;
      if (!ready || clock.seeking) return false;
      return [0, 1, 2].includes(player.getPlayerState());
    },
    get playbackRate() { return local ? nativeVideo.playbackRate : ready ? player.getPlaybackRate() || 1 : 1; },
    set playbackRate(rate) { if (local) nativeVideo.playbackRate = rate; else if (ready) player.setPlaybackRate(rate); },
    async play() {
      if (clock.currentTime >= duration - .03) clock.currentTime = 0;
      if (local) return nativeVideo.play();
      if (ready) player.playVideo();
    },
    pause() { if (local) nativeVideo.pause(); else if (ready && !clock.paused) player.pauseVideo(); },
    get local() { return local; },
  };
  fallback.onclick = () => switchToLocal('Playing the local copy.');
  status.textContent = 'Loading YouTube…';

  function setup() {
    if (local) return;
    player = new window.YT.Player('youtube-player', {
      videoId: source.id,
      playerVars: {playsinline: 1, autoplay: 0, rel: 0, origin: location.origin, end: duration},
      events: {
        onReady() {
          if (local) return;
          ready = true;
          status.textContent = '';
          if (requestedTime) clock.currentTime = requestedTime;
          onChange();
        },
        onError() { if (!local) status.textContent = 'YouTube playback is unavailable here. Use the local video.'; },
        onStateChange() { if (!local) onChange(); },
      },
    });
  }
  if (window.YT?.Player) setup();
  else {
    window.onYouTubeIframeAPIReady = setup;
    const script = document.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    script.async = true;
    script.onerror = () => { if (!local) status.textContent = 'YouTube did not load. Use the local video.'; };
    document.head.append(script);
    setTimeout(() => {
      if (!ready && !local) status.textContent = 'YouTube is taking longer to load. You can use the local video.';
    }, 9000);
  }
  return clock;
}

export function attachMatchTimeline(data, seek) {
  const timeline = document.getElementById('match-overview');
  const rallyState = document.getElementById('match-rally-state');
  const duration = data.clip.end;
  const rallies = data.reviewed_rallies;
  const analyzed = rallies.filter(rally => rally.section != null);
  const clockText = time => {
    const tenths = Math.round(time * 10);
    const seconds = (tenths % 600 / 10).toFixed(1).padStart(4, '0');
    return `${Math.floor(tenths / 600)}:${seconds}`;
  };
  const lanes = [
    [document.getElementById('rally-play-lane'), rallies, 'Ball in play'],
    [document.getElementById('review-lane'), analyzed, 'Analyzed'],
  ];
  for (const [lane, spans, label] of lanes) {
    lane.addEventListener('click', event => {
      const bounds = lane.getBoundingClientRect();
      const fraction = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
      seek(fraction * duration);
    });
    for (const {start,end,approximate} of spans) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'match-span';
      button.style.left = `${100 * start / duration}%`;
      button.style.width = `${100 * (end-start) / duration}%`;
      button.title = `${label} · ${clockText(start)}–${clockText(end)}${approximate ? ' (approximate)' : ''}`;
      button.setAttribute('aria-label', button.title);
      button.onclick = event => {
        if (event.detail !== 0) return;
        event.stopPropagation();
        seek(Math.min(start + .04, (start + end) / 2));
      };
      lane.append(button);
    }
  }
  return time => {
    timeline.style.setProperty('--match-playhead', `${100 * Math.min(duration, time) / duration}%`);
    const rally = rallies.find(({start,end}) => start <= time && time <= end);
    let text = 'Rally status unreviewed here';
    if (rally) text = rally.approximate ? 'Ball in play · approximate boundaries' : 'Ball in play';
    else if (data.rally_reviewed_through > 0 && time <= data.rally_reviewed_through) text = 'Between rallies';
    if (rallyState.textContent !== text) rallyState.textContent = text;
    return analyzed.find(({start,detail_end}) => start <= time && time < detail_end)?.section;
  };
}
