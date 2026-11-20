"use strict";
const match = window.MATCH;
const tracks = window.TRACKS;
const review = window.MATCH_REVIEW;
const possession = match.possession;
const reviewedObservations = review?.possession?.observations.map((row) => [row.seconds, row]) ?? [];
const players = new Map(match.people.map((player) => [player.id, player]));
const $ = (id) => document.getElementById(id);
const phases = new Set(["kickoff", "halftime", "second_half", "full_time", "phase"]);
const pending = new Set(["candidate", "unresolved"]);
const goalLeadSeconds = 5;
const teamOrder = match.teamOrder ?? ["team_1", "team_2"];
const teamClass = {team_1: "team-1", team_2: "team-2"};
const teamColor = {team_1: "#a8f1bc", team_2: "#9faeff", ...match.teamColors};
const position = (seconds) => `${Math.max(0, Math.min(100, seconds / match.duration * 100))}%`;
const formatTime = (seconds) => {
  const value = Math.floor(Math.max(0, seconds));
  const remainder = String(value % 60).padStart(2, "0");
  return `${Math.floor(value / 60)}:${remainder}`;
};
const canvas = $("overlay");
const showOverlay = $("show-overlay");
const clock = $("clock");
const scrub = $("scrub");
const context = canvas?.getContext("2d");
const eventList = $("event-list");
const rail = $("event-rail");
const eventMarkers = $("event-markers");
const activeEvents = [];
const laneElements = [];
let readTime = () => seekTime;
let readFrameTime = () => readTime();
let writeTime;
let playVideo;
let playRequested = false;
let seekTime = match.gamePeriods[0]?.[0] ?? match.trackedPeriods[0]?.[0] ?? 0;
let selectedPlayer = null;
let scrubbing = false;
let drawnFrame = null;
let previousFocus;
let previousOverlay;
let previousOwner;
let previousPossession;

function element(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text) result.textContent = text;
  return result;
}

function seek(seconds, {play = false} = {}) {
  seekTime = Math.max(0, Math.min(match.duration, seconds));
  playRequested ||= play;
  if (writeTime) writeTime(seekTime, playRequested);
  if (playRequested && playVideo) {
    playRequested = false;
    playVideo();
  }
  drawTime(seekTime);
}

function media() {
  const player = $("player");
  if (!match.local || (match.videoId && !review)) {
    youtubeMedia(player);
    return;
  }
  const video = element("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.src = match.mediaFile || "video.mp4";
  let presentedTime = null;
  const hasFrameCallback = typeof video.requestVideoFrameCallback === "function";
  readFrameTime = () => {
    if (video.seeking || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return null;
    return hasFrameCallback ? presentedTime : video.currentTime;
  };
  if (hasFrameCallback) {
    function presented(_now, frame) {
      presentedTime = frame.mediaTime;
      video.requestVideoFrameCallback(presented);
    }
    video.requestVideoFrameCallback(presented);
  }
  video.addEventListener("seeking", () => {
    presentedTime = null;
    drawBoxes(null);
  });
  video.addEventListener("loadedmetadata", () => {
    readTime = () => video.currentTime;
    writeTime = (time) => { video.currentTime = time; };
    playVideo = () => { video.play().catch(() => {}); };
    seek(seekTime);
  });
  video.addEventListener("error", () => {
    if (match.videoId && location.protocol.startsWith("http")) {
      video.remove();
      writeTime = undefined;
      playVideo = undefined;
      readTime = () => seekTime;
      readFrameTime = () => readTime();
      youtubeMedia(player);
    } else {
      $("media-status").textContent = "Local recording unavailable.";
    }
  }, {once: true});
  player.append(video);
}

function youtubeMedia(player) {
  if (!match.videoId) return;
  const initialTime = seekTime;
  const options = new URLSearchParams({
    enablejsapi: 1, autoplay: 0, playsinline: 1, rel: 0, start: Math.floor(initialTime),
  });
  if (location.protocol.startsWith("http")) options.set("origin", location.origin);
  const frame = element("iframe");
  frame.title = `${match.title} on YouTube`;
  frame.allowFullscreen = true;
  frame.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
  frame.referrerPolicy = "strict-origin-when-cross-origin";
  frame.src = `https://www.youtube.com/embed/${encodeURIComponent(match.videoId)}?${options}`;
  player.append(frame);
  window.onYouTubeIframeAPIReady = () => {
    new YT.Player(frame, {
      events: {
        onReady({target}) {
          const waitingToPlay = () => {
            const state = target.getPlayerState();
            return state === YT.PlayerState.UNSTARTED || state === YT.PlayerState.CUED;
          };
          readTime = () => waitingToPlay() ? seekTime : target.getCurrentTime() ?? seekTime;
          writeTime = (time, play) => {
            if (!play && waitingToPlay()) target.cueVideoById({videoId: match.videoId, startSeconds: time});
            else target.seekTo(time, true);
          };
          playVideo = () => target.playVideo();
          $("media-status").textContent = "";
          // The iframe is already cued at kickoff. Seeking here can start playback.
          if (playRequested || seekTime !== initialTime) seek(seekTime);
        },
        onError() {
          $("media-status").textContent = location.protocol === "file:"
            ? "Open the preview over HTTP for YouTube playback."
            : "Embedded YouTube playback is unavailable.";
        },
      },
    });
  };
  const script = element("script");
  script.src = "https://www.youtube.com/iframe_api";
  script.onerror = () => { $("media-status").textContent = "YouTube timeline controls could not connect."; };
  document.head.append(script);
}

function decorateRail(parent) {
  const periods = match.gamePeriods.length ? match.gamePeriods : match.trackedPeriods;
  for (const [start, end] of periods) {
    const item = element("span", "period");
    item.style.left = position(start);
    item.style.width = position(end - start);
    parent.append(item);
  }
  parent.append(element("span", "playline"));
}

function layoutGoalMarkers() {
  const goalEnds = [];
  const width = rail.clientWidth;
  for (const {time, marker} of activeEvents) {
    if (!marker.classList.contains("goal")) continue;
    const x = time / match.duration * width;
    const available = goalEnds.findIndex((end) => end <= x);
    const row = available < 0 ? goalEnds.length : available;
    goalEnds[row] = x + marker.offsetWidth + 2;
    marker.style.setProperty("--goal-row", row);
  }
  rail.classList.toggle("has-goals", goalEnds.length > 0);
  rail.style.setProperty("--goal-rows", goalEnds.length);
}

function events() {
  eventList?.replaceChildren();
  eventMarkers.replaceChildren();
  decorateRail(eventMarkers);
  activeEvents.length = 0;
  const filter = $("event-filter")?.value ?? "goals";
  $("event-rail-label").textContent = eventList ? "Events" : "Goals";
  for (const event of review?.events ?? match.events) {
    const open = pending.has(event.status);
    const goal = event.type === "goal";
    const phase = phases.has(event.type);
    if (filter === "key" && (!goal && !phase || open && !phase)) continue;
    if (filter === "all" && open || filter === "goals" && (!goal || open) ||
        filter === "restarts" && event.type !== "restart" && !event.restart_seconds ||
        filter === "open" && !open) continue;
    const label = event.label + (open ? " · open" : "");
    const style = open ? "candidate" : goal ? "goal" : "";
    let button;
    if (eventList) {
      const row = element("div", "event-row");
      button = element("button", `event-button ${style}`);
      button.type = "button";
      button.append(element("time", "event-time", formatTime(event.seconds)),
                    element("span", "event-detail", label));
      button.addEventListener("click", () => seek(event.seconds));
      row.append(button);
      if (event.restart_seconds != null) {
        const restart = element("button", "restart-button", "Restart ↗");
        restart.type = "button";
        restart.title = `Seek to restart at ${formatTime(event.restart_seconds)}`;
        restart.addEventListener("click", () => seek(event.restart_seconds));
        row.append(restart);
      }
      eventList.append(row);
    }
    const marker = element("button", `marker ${style}`);
    marker.type = "button";
    marker.style.left = position(event.seconds);
    marker.title = `${formatTime(event.seconds)} · ${label}`;
    if (goal && !open) {
      const ball = element("span", "goal-ball", "⚽");
      ball.setAttribute("aria-hidden", "true");
      const color = teamColor[match.goalTeams?.[event.id]];
      if (color) ball.style.setProperty("--goal-color", color);
      marker.append(ball);
      const score = match.goalScores?.[event.id];
      if (score) {
        const result = element("span", "goal-score", score.join("–"));
        result.setAttribute("aria-hidden", "true");
        marker.append(result);
        marker.title += ` · ${teamOrder.map((team, index) => `${match.teamNames[team]} ${score[index]}`).join(", ")}`;
      }
    }
    marker.setAttribute("aria-label", marker.title);
    marker.addEventListener("click", () => {
      if (goal && !open) {
        seek(event.seconds - goalLeadSeconds, {play: true});
      } else {
        seek(event.seconds);
      }
    });
    eventMarkers.append(marker);
    activeEvents.push({time: event.seconds, button, marker});
  }
  layoutGoalMarkers();
  if (eventList && !eventList.childElementCount) eventList.append(element("p", "muted", "No events in this view."));
}

function focusPlayer(player) {
  selectedPlayer = player;
  $("focus-label").textContent = player ? `Following ${player.label}` : "";
  $("clear-focus").hidden = !player;
  for (const item of laneElements) {
    item.row.toggleAttribute("data-focused", item.player === player);
    item.button.setAttribute("aria-pressed", item.player === player ? "true" : "false");
  }
  if (player) {
    showOverlay.checked = true;
    canvas.hidden = false;
    seek(player.first);
  }
}

function roster() {
  const lanes = $("lanes");
  if (!match.people.length) {
    lanes.append(element("p", "muted", "No player observations published for this recording."));
    return;
  }
  const tabs = $("team-tabs");
  const teams = [...teamOrder];
  if (review && match.people.some((player) => !teamClass[player.team])) teams.push("unknown");
  const views = [];
  function showTeam(view) {
    if (selectedPlayer && (selectedPlayer.team || "unknown") !== view.team) focusPlayer(null);
    for (const item of views) {
      const selected = item === view;
      item.tab.setAttribute("aria-selected", String(selected));
      item.tab.tabIndex = selected ? 0 : -1;
      item.group.hidden = !selected;
    }
  }
  for (const team of teams) {
    const people = match.people.filter((player) => (player.team || "unknown") === team);
    const color = teamClass[team] || "team-unknown";
    const group = element("section", `team ${color}`);
    const tab = element("button", `team-tab ${color}`, match.teamNames[team]);
    if (teamColor[team]) {
      tab.style.setProperty("--team-color", teamColor[team]);
      group.style.setProperty("--team-color", teamColor[team]);
    }
    tab.type = "button";
    tab.id = `tab-${team}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `lanes-${team}`);
    group.id = `lanes-${team}`;
    group.setAttribute("role", "tabpanel");
    group.setAttribute("aria-labelledby", tab.id);
    group.tabIndex = 0;
    const view = {team, tab, group};
    views.push(view);
    tab.addEventListener("click", () => showTeam(view));
    tabs.append(tab);
    if (!people.length) group.append(element("p", "muted", "No player observations for this team."));
    for (const player of people) {
      const row = element("div", "person-lane");
      const button = element("button", "person-select");
      button.type = "button";
      button.title = showOverlay ? `Follow ${player.label}` : "Play from first appearance";
      button.setAttribute("aria-label", showOverlay ? button.title : `Play from ${player.label}'s first appearance`);
      if (showOverlay) button.setAttribute("aria-pressed", "false");
      const photo = element(player.portrait ? "img" : "span", player.portrait ? "" : "placeholder");
      if (player.portrait) {
        photo.src = player.portrait;
        photo.loading = "lazy";
        photo.alt = "";
      }
      button.append(photo);
      if (review) {
        button.append(element("span", "", player.label));
        if (player.role === "goalkeeper") button.append(element("small", "", "GK"));
      }
      button.addEventListener("click", () => {
        if (showOverlay) focusPlayer(selectedPlayer === player ? null : player);
        else seek(player.first, {play: true});
      });
      const bar = element("button", "rail player-rail");
      bar.type = "button";
      bar.setAttribute("aria-label", `Play from ${player.label}'s tracking lane`);
      decorateRail(bar);
      for (const [start, end] of player.spans) {
        const span = element("span", "appearance");
        span.style.left = position(start);
        span.style.width = position(end - start);
        bar.append(span);
      }
      bar.addEventListener("click", (event) => {
        if (!event.detail) {
          seek(player.first, {play: true});
          return;
        }
        const bounds = bar.getBoundingClientRect();
        seek((event.clientX - bounds.left) / bounds.width * match.duration, {play: true});
      });
      row.append(button, bar);
      if (review) {
        const count = element("span", "observed", formatTime(player.observed));
        count.title = "Time visible in tracked frames";
        const controlledSeconds = possession?.summary.players[player.id];
        if (controlledSeconds) count.title += ` · ${formatTime(controlledSeconds)} identified possession`;
        row.append(count);
      }
      group.append(row);
      laneElements.push({player, row, button});
    }
    lanes.append(group);
  }
  tabs.addEventListener("keydown", (event) => {
    const index = views.findIndex((view) => view.tab === event.target);
    const next = {Home: 0, End: views.length - 1, ArrowLeft: (index + views.length - 1) % views.length,
                  ArrowRight: (index + 1) % views.length}[event.key];
    if (index < 0 || next === undefined) return;
    event.preventDefault();
    showTeam(views[next]);
    views[next].tab.focus();
  });
  showTeam(views.find((view) => view.group.querySelector(".person-lane")) ?? views[0]);
  tabs.hidden = false;
}

function indexAt(rows, time) {
  let lo = 0;
  let hi = rows.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (rows[mid][0] <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

function currentPossession(time) {
  return possession?.intervals[indexAt(possession.intervals, time)];
}

const playColors = {live: "#67cb9d", loose: "#67cb9d", dead: "#e2a969", unknown: "#596965", outside: "#111b1a"};
const playLabels = {live: "In play", loose: "In play · loose ball", dead: "Out of play",
                    unknown: "Play state uncertain", outside: "Outside match"};
const exactTime = (time) => `${formatTime(time)}.${String(Math.floor(time % 1 * 1000 + 1e-6)).padStart(3, "0")}`;

function possessionLabel(row) {
  if (!row) return "";
  const player = players.get(row[3]);
  if (player) return [player.label, match.teamNames[player.team]].filter(Boolean).join(" · ");
  return row[2] === "live" ? "In play · player unknown" : playLabels[row[2]];
}

function plotRail(rail, paint) {
  const chart = element("canvas", "rail-chart");
  chart.setAttribute("aria-hidden", "true");
  rail.append(chart, element("span", "playline"));
  const resize = () => {
    const bounds = rail.getBoundingClientRect();
    chart.width = Math.max(1, Math.round(bounds.width * devicePixelRatio));
    chart.height = Math.max(1, Math.round(bounds.height * devicePixelRatio));
    paint(chart.getContext("2d"), chart.width, chart.height);
  };
  new ResizeObserver(resize).observe(rail);
}

function seekRail(rail, title) {
  const moment = (event) => {
    const bounds = rail.getBoundingClientRect();
    return Math.max(0, Math.min(match.duration, (event.clientX - bounds.left) / bounds.width * match.duration));
  };
  rail.addEventListener("pointermove", (event) => rail.title = title(moment(event)));
  rail.addEventListener("click", (event) => seek(event.detail ? moment(event) : readTime()));
}

function possessionPanel() {
  if (!possession) return;
  $("possession-now").hidden = false;
  $("possession-lanes").hidden = false;
  for (const [id, playState] of [["play-state-rail", true], ["possession-rail", false]]) {
    const rail = $(id);
    plotRail(rail, (ctx, width, height) => {
      for (const [start, end, state, owner] of possession.intervals) {
        const player = players.get(owner);
        ctx.fillStyle = playState ? playColors[state] :
          player ? teamColor[player.team] || "#b5c7be" : state === "outside" ? playColors.outside : "#283632";
        ctx.fillRect(start / match.duration * width, 0, (end - start) / match.duration * width, height);
      }
    });
    seekRail(rail, (time) => {
      const row = currentPossession(time);
      return `${formatTime(time)} · ${playState ? playLabels[row?.[2]] : possessionLabel(row)}`;
    });
  }
  const legend = $("possession-legend");
  for (const state of ["live", "dead", "unknown"]) {
    const item = element("span", "legend-item", playLabels[state]);
    item.style.setProperty("--swatch", playColors[state]);
    legend.append(item);
  }
  if (!possession.hasEstimate && !possession.coverage) return;
  $("possession-card").hidden = false;
  const summary = possession.summary;
  if (possession.coverage && !possession.coverage.ready) {
    const remaining = possession.coverage.unresolvedSeconds + (possession.coverage.sparseReviewSeconds ?? 0);
    $("possession-description").textContent =
      `Possession review incomplete. ${formatTime(remaining)} of ${formatTime(summary.matchSeconds)} still needs footage review.`;
    $("possession-share").hidden = true;
    return;
  }
  const counts = $("possession-teams");
  const bar = $("possession-share");
  const percent = (value) => `${value.toFixed(value > 0 && value < 1 ? 2 : 1)}%`;
  for (const team of ["team_1", "team_2"]) {
    const stats = summary.teams[team];
    const color = teamColor[team];
    const item = element("div", "team-possession");
    item.style.setProperty("--swatch", color);
    item.append(element("strong", "", stats.matchPercent == null ? "—" : percent(stats.matchPercent)),
                element("span", "", `${match.teamNames[team]} · share of match time`));
    if (stats.identifiedPercent != null) {
      item.append(element("small", "muted", `${percent(stats.identifiedPercent)} of identified possession`));
    }
    counts.append(item);
    const segment = element("span");
    segment.style.width = `${stats.identifiedPercent ?? 0}%`;
    segment.style.background = color;
    bar.append(segment);
  }
  $("possession-description").textContent =
    `Player in control identified for ${formatTime(summary.assignedSeconds)} of ${formatTime(summary.matchSeconds)}. ` +
    `Play state uncertain for ${formatTime(summary.unknownSeconds)}.`;
  bar.setAttribute("aria-hidden", "true");
}

function nearestFrame(time) {
  const positions = tracks.times;
  const ms = time * 1000;
  let lower = 0;
  let upper = positions.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (positions[middle] < ms) lower = middle + 1;
    else upper = middle;
  }
  if (lower > 0 && (lower === positions.length || ms - positions[lower - 1] < positions[lower] - ms)) lower--;
  return Math.abs(positions[lower] - ms) <= 520 / match.fps ? lower : null;
}

function drawBoxes(time) {
  if (!context) return;
  const showing = showOverlay.checked;
  const frame = showing && time != null ? nearestFrame(time) : null;
  const owner = time == null ? null : currentPossession(time)?.[3];
  if (frame === drawnFrame && showing === previousOverlay && previousFocus === selectedPlayer &&
      owner === previousOwner) return;
  drawnFrame = frame;
  previousOverlay = showing;
  previousFocus = selectedPlayer;
  previousOwner = owner;
  context.clearRect(0, 0, canvas.width, canvas.height);
  const size = Math.max(12, canvas.width / 100);
  context.font = `600 ${size}px system-ui`;
  context.lineWidth = Math.max(2, canvas.width / 640);
  for (const [index, x1, y1, x2, y2, reviewed] of tracks.frames[frame] ?? []) {
    const player = match.people[index];
    if (selectedPlayer && player !== selectedPlayer) continue;
    const color = teamColor[player.team] || "#f0f6ee";
    const label = player.label + (owner === player.id ? " ●" : "") + (review && reviewed ? " ✓" : "");
    const width = context.measureText(label).width + size * .6;
    const top = Math.max(size + 1, y1);
    context.strokeStyle = color;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.fillStyle = "#0c1515";
    context.fillRect(x1, top - size * 1.1, width, size * 1.1);
    context.fillStyle = color;
    context.fillText(label, x1 + size * .3, top - size * .16);
  }

}

function drawTime(time) {
  const timeLabel = formatTime(time);
  if (clock && clock.textContent !== timeLabel) clock.textContent = timeLabel;
  if (scrub.getAttribute("aria-valuetext") !== timeLabel) scrub.setAttribute("aria-valuetext", timeLabel);
  if (!scrubbing) scrub.value = time;
  $("lanes").parentElement.style.setProperty("--playhead", position(time));
  $("reviews")?.style.setProperty("--playhead", position(time));
  const current = currentPossession(time);
  if (current !== previousPossession) {
    $("possession-now").textContent = possessionLabel(current);
    const player = players.get(current?.[3]);
    $("possession-now").style.color = player ? teamColor[player.team] : "";
    previousPossession = current;
  }
  if (review?.possession && $("possession-evidence")) {
    const evidence = review.possession;
    const at = reviewedObservations[indexAt(reviewedObservations, time + .04)]?.[1];
    const decision = evidence.results[indexAt(evidence.results, time)];
    const label = at && Math.abs(at.seconds - time) <= .1
      ? `${exactTime(at.seconds)} · ${playLabels[at.state]}${at.player ? ` · ${players.get(at.player)?.label}` : ""}${at.evidence ? ` · ${at.evidence}` : ""}`
      : decision?.[4] === "review" ? "Visually reviewed interval" :
        decision?.[4] === "event" ? "Reviewed event and restart" : "No visual possession decision at this time.";
    if ($("possession-evidence").textContent !== label) $("possession-evidence").textContent = label;
  }
  for (const item of activeEvents) {
    const active = Math.abs(item.time - time) <= 1;
    for (const control of [item.button, item.marker]) {
      if (!control) continue;
      if (active) control.setAttribute("aria-current", "true");
      else if (control.hasAttribute("aria-current")) control.removeAttribute("aria-current");
    }
  }
  drawBoxes(readFrameTime());
}

function tick() {
  if (!scrubbing && !document.hidden) drawTime(readTime());
  requestAnimationFrame(tick);
}

function reviewTicks(parent, label, times, color) {
  if (!times.length) return;
  const row = element("div", "axis");
  const rail = element("button", "rail state-rail");
  rail.type = "button";
  rail.setAttribute("aria-label", `Seek to the closest ${label.toLowerCase()} frame`);
  row.append(element("span", "lane-label", label), rail, element("span", "observed", String(times.length)));
  parent.append(row);
  plotRail(rail, (ctx, width, height) => {
    ctx.fillStyle = color;
    for (const time of times) ctx.fillRect(Math.round(time / match.duration * (width - 1)), 0, 1, height);
  });
  const closest = (time) => {
    let low = 0;
    let high = times.length;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (times[mid] < time) low = mid + 1;
      else high = mid;
    }
    const before = times[low - 1];
    const after = times[low];
    return after == null || time - before < after - time ? before : after;
  };
  rail.addEventListener("pointermove", (event) => {
    const rect = rail.getBoundingClientRect();
    rail.title = exactTime(closest((event.clientX - rect.left) / rect.width * match.duration));
  });
  rail.addEventListener("click", (event) => {
    const rect = rail.getBoundingClientRect();
    seek(closest(event.detail ? (event.clientX - rect.left) / rect.width * match.duration : readTime()));
  });
}

function download(href, label) {
  const link = element("a", "", label);
  link.href = href;
  link.download = "";
  return link;
}

function addSheet(parent, src, description) {
  const image = element("img", "crop-sheet");
  image.src = src;
  image.alt = description;
  image.loading = "lazy";
  const link = element("a", "crop-link");
  link.href = src;
  link.target = "_blank";
  link.rel = "noopener";
  link.append(image);
  parent.append(link);
}

function addFrameSeeks(parent, cases) {
  const buttons = element("div", "crop-seeks");
  for (const [stamp, name, state] of cases) {
    const button = element("button", state === "unresolved" ? "held" : "", exactTime(stamp));
    button.type = "button";
    button.title = state ? `${name} · ${state}` : `Tile ${name}`;
    button.addEventListener("click", () => {
      seek(stamp);
      $("viewport").scrollIntoView({block: "center"});
    });
    buttons.append(button);
  }
  parent.append(buttons);
}

function reviewPanel() {
  if (!review) return;
  const section = $("reviews");
  section.append(element("h2", "", "Tracking review"));
  if (review.eventCoverage) section.append(element("p", "muted", review.eventCoverage));
  if (review.heldObservations) {
    section.append(element("p", "muted", `${review.heldObservations} player observations held from the overlay.`));
  }
  const files = element("div", "review-files");
  files.append(download("review/data.json", "Review data"));
  if (tracks.times.length) files.append(download("observations.csv.gz", "Tracking observations"));
  if (review.crops?.length) files.append(download("review/crops/identity-audit.json", "Crop frame index"));
  section.append(files);

  reviewTicks(section, "Events inspected", review.eventChecks, "#ffcc86");
  reviewTicks(section, "Player crops inspected", review.identityReviewChecks ?? [], "#b3d8ff");
  reviewTicks(section, "Possession inspected", review.possession?.checked ?? [], "#bca9ff");
  reviewTicks(section, "Tracking edits", review.identityChecks ?? [], "#94d8ff");
  const packs = review.possession?.packs ?? [];
  const packed = packs.flatMap((pack) => Object.values(pack.sheets).flatMap((rows) => rows.map(([time]) => time)));
  reviewTicks(section, "Possession sheets", [...new Set(packed)].sort((a, b) => a - b), "#6e9184");
  if (review.possession) {
    const evidence = element("output", "evidence");
    evidence.id = "possession-evidence";
    section.append(evidence);
  }
  if (review.possession?.cadenceReview?.note) {
    section.append(element("p", "muted", review.possession.cadenceReview.note));
  }
  const queues = [
    [review.possession?.coverageGaps ?? [], "unresolved gaps of at least one second"],
    [review.possession?.cadenceGaps ?? [], "gaps between checked frames longer than half a second"],
  ];
  for (const [gaps, label] of queues) {
    if (!gaps.length) continue;
    const details = element("details", "crop-review");
    details.append(element("summary", "", `${gaps.length} ${label}`));
    for (const [start, end] of gaps) {
      const row = element("p", "review-note");
      const button = element("button", "", `${exactTime(start)}–${exactTime(end)}`);
      button.type = "button";
      button.addEventListener("click", () => seek(start));
      row.append(button);
      details.append(row);
    }
    section.append(details);
  }
  if (packs.length) {
    section.append(element("h3", "crop-heading", "Possession source sheets"));
    for (const pack of packs) {
      const group = element("details", "crop-review");
      const sheets = Object.entries(pack.sheets);
      const count = sheets.reduce((sum, [, frames]) => sum + frames.length, 0);
      group.append(element("summary", "", `${pack.fps} FPS · ${count} source frames`));
      group.addEventListener("toggle", () => {
        if (!group.open || group.childElementCount > 1) return;
        group.append(download(pack.index, "Exact frame and player IDs"));
        for (const [src, frames] of sheets) {
          const sheet = element("details", "crop-review");
          sheet.append(element("summary", "", `${exactTime(frames[0][0])} · ${frames.length} frames`));
          sheet.addEventListener("toggle", () => {
            if (!sheet.open || sheet.childElementCount > 1) return;
            addSheet(sheet, src, "Source video with tracked player labels and crops");
            addFrameSeeks(sheet, frames);
          });
          group.append(sheet);
        }
      });
      section.append(group);
    }
  }

  section.append(element("h3", "crop-heading", "Player crops across the match"));
  if (!review.crops?.length) section.append(element("p", "muted", "No chronological crop sheets supplied."));
  for (const player of review.crops ?? []) {
    const group = element("details", "crop-review");
    const sheets = Object.entries(player.sheets);
    const count = sheets.reduce((sum, [, cases]) => sum + cases.length, 0);
    group.append(element("summary", "", `${player.label} · ${count} crops`));
    group.addEventListener("toggle", () => {
      if (!group.open || group.childElementCount > 1) return;
      addSheet(group, player.reference, `${player.label} reference crops`);
      for (const [sheet, cases] of sheets) {
        addSheet(group, sheet, `${player.label} crops from ${exactTime(cases[0][0])} to ${exactTime(cases.at(-1)[0])}`);
        addFrameSeeks(group, cases);
      }
    });
    section.append(group);
  }
  const notes = [...review.reviews, ...(review.possession?.intervals ?? []).map((interval) => ({
    seconds: interval.start, label: interval.evidence,
  }))];
  if (notes.length) {
    const details = element("details", "crop-review");
    details.append(element("summary", "", `Review notes · ${notes.length}`));
    for (const item of notes) {
      const start = typeof item.seconds === "number" ? item.seconds :
        item.seconds.split(":").reduce((result, part) => result * 60 + Number(part), 0);
      const paragraph = element("p", "review-note");
      const button = element("button", "", exactTime(start));
      button.type = "button";
      button.addEventListener("click", () => seek(start));
      paragraph.append(button, document.createTextNode(` ${item.label}`));
      details.append(paragraph);
    }
    section.append(details);
  }
}

document.title = match.title + (review ? " · Developer review" : "");
const matchTitle = $("match-title");
const dreamTeamName = /\bDream Team\b/i.exec(match.title);
if (dreamTeamName) {
  const [name] = dreamTeamName;
  const team = teamOrder.find((id) => match.teamNames[id] === "Dream Team");
  const highlightedName = element("span", "", name);
  highlightedName.style.color = match.teamColors?.[team] ?? "#ff93bb";
  const start = dreamTeamName.index;
  matchTitle.append(match.title.slice(0, start), highlightedName, match.title.slice(start + name.length));
} else {
  matchTitle.textContent = match.title;
}
$("match-date").textContent = match.date || "";
$("match-score").textContent = match.score || "";
$("viewport").style.setProperty("--video-width", match.width);
$("viewport").style.setProperty("--video-height", match.height);
if (review && match.people.length) $("coverage").textContent = "Player lanes show when each player is visible.";
$("clear-focus").addEventListener("click", () => focusPlayer(null));
if (showOverlay) {
  canvas.width = match.width;
  canvas.height = match.height;
  $("overlay-label").hidden = !match.people.length;
  showOverlay.addEventListener("change", () => {
    canvas.hidden = !showOverlay.checked;
    drawBoxes(readFrameTime());
  });
}
$("event-filter")?.addEventListener("change", events);
scrub.max = match.duration;
scrub.addEventListener("input", () => {
  scrubbing = true;
  drawTime(Number(scrub.value));
});
scrub.addEventListener("change", () => {
  scrubbing = false;
  seek(Number(scrub.value));
});
const step = match.duration > 1800 ? 600 : match.duration > 300 ? 120 : match.duration > 90 ? 30 : 10;
const labels = [0];
for (let time = step; time + step * .28 < match.duration; time += step) labels.push(time);
labels.push(match.duration);
for (const time of labels) {
  const item = element("span", "tick", formatTime(time));
  item.style.left = position(time);
  $("ticks").append(item);
}
reviewPanel();
possessionPanel();
events();
new ResizeObserver(layoutGoalMarkers).observe(rail);
roster();
media();
drawTime(seekTime);
requestAnimationFrame(tick);
