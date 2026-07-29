#!/usr/bin/env bash
# Download SemanticKITTI (velodyne point clouds + panoptic labels) into $1.
#
#   bash scripts/download_semantickitti.sh /data/semantickitti
#
# Produces:  $DEST/dataset/sequences/{00..21}/velodyne/*.bin
#            $DEST/dataset/sequences/{00..10}/labels/*.label   (train+val only)
# Point the config at it:  configs/data/semantickitti.yaml -> root: $DEST/dataset
#
# Sizes: velodyne ~80GB (KITTI odometry), labels ~180MB. Need ~170GB free while unzipping
# (zip + extracted coexist). Both URLs below are the standard public mirrors.
set -euo pipefail

DEST="${1:?usage: download_semantickitti.sh <dest-dir>}"
mkdir -p "$DEST"
cd "$DEST"

VELO_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
LABEL_URL="http://www.semantic-kitti.org/assets/data_odometry_labels.zip"

avail=$(df -Pk "$DEST" | awk 'NR==2{print int($4/1024/1024)}')
echo "==> free at $DEST: ${avail}GB (need ~170GB peak during unzip)"

fetch() {  # url, out
  if [ -f "$2" ]; then echo "==> $2 exists, skip download"; else
    echo "==> downloading $2"; wget -c -O "$2" "$1"; fi
}

fetch "$LABEL_URL" data_odometry_labels.zip
fetch "$VELO_URL"  data_odometry_velodyne.zip

echo "==> unzip labels"; unzip -n -q data_odometry_labels.zip
echo "==> unzip velodyne (large, minutes)"; unzip -n -q data_odometry_velodyne.zip

echo "==> layout check"
for seq in 00 08 10; do
  v=$(ls "dataset/sequences/$seq/velodyne"/*.bin 2>/dev/null | wc -l)
  l=$(ls "dataset/sequences/$seq/labels"/*.label 2>/dev/null | wc -l)
  echo "   seq $seq: $v scans, $l labels"
done
echo "==> done. set configs/data/semantickitti.yaml root: $DEST/dataset"
echo "    (optional) rm data_odometry_*.zip to reclaim ~80GB"
