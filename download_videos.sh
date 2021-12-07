#!/usr/bin/env bash

input_file=$1
output_folder=$2

mapfile -t video_ids < <(jq --raw-output '.[] | .[] | .video' "$input_file")

n=${#video_ids[@]}

# Pairs of video and audio URLs.
mapfile -t urls < <(youtube-dl -g -a <(
  IFS=$'\n'
  echo "${video_ids[*]}"
) |
  tqdm --total "$((2 * n))" --desc "Getting URLs")

mapfile -t start_times < <(jq --raw-output '.[] | .[] | .time_s' "$input_file")
mapfile -t end_times < <(jq --raw-output '.[] | .[] | .time_e' "$input_file")

# From https://unix.stackexchange.com/a/426827:

# converts HH:MM:SS.sss to fractional seconds
codes2seconds() (
  local hh=${1%%:*}
  local rest=${1#*:}
  local mm=${rest%%:*}
  local ss=${rest#*:}
  printf "%s" "$(bc <<<"$hh * 60 * 60 + $mm * 60 + $ss")"
)

# converts fractional seconds to HH:MM:SS.sss
seconds2codes() (
  local seconds=$1
  local hh
  hh=$(bc <<<"scale=0; $seconds / 3600")
  local remainder
  remainder=$(bc <<<"$seconds % 3600")
  local mm
  mm=$(bc <<<"scale=0; $remainder / 60")
  local ss
  ss=$(bc <<<"$remainder % 60")
  printf "%02d:%02d:%06.3f" "$hh" "$mm" "$ss"
)

subtract_times() (
  local t1sec
  t1sec=$(codes2seconds "$1")
  local t2sec
  t2sec=$(codes2seconds "$2")
  printf "%s" "$(bc <<<"$t2sec - $t1sec")"
)

for i in "${!video_ids[@]}"; do
  video_id=${video_ids[$i]}
  video_url=${urls[$((2 * i))]}
  audio_url=${urls[$((2 * i + 1))]}
  start_time=${start_times[$i]}
  end_time=${end_times[$i]}
  duration=$(subtract_times "$start_time" "$end_time")

  # TODO: start from previous key frame
  # TODO: lower the quality?
  # -fs 25M \
  # -avoid_negative_ts 1 \
  # Note "-to" doesn't work if "-ss" is before the video, which should to be.
  # So using "-t".
  ffmpeg \
    -ss "$start_time" \
    -i "$video_url" \
    -ss "$start_time" \
    -i "$audio_url" \
    -map 0:v \
    -map 1:a \
    -t "$duration" \
    -c:v copy \
    -c:a copy \
    -n \
    "$output_folder/$video_id+${start_time%%.*}+${end_time%%.*}.mp4" >/dev/null 2>&1
    # Note this uses floor to round the durations in the filename, not round.

  echo "$i"
done | tqdm --total "$n" --desc "Downloading videos" >/dev/null
