#!/bin/bash

# -------------- CONFIG --------------
GITHUB_USER="jameszianxuTT"       # CHANGE THIS
REPO="logdump"                # CHANGE THIS
BRANCH="main"
# -----------------------------------

# -------------- INPUT --------------
LOG_PATH="$1"
BASENAME="$2"

if [ -z "$LOG_PATH" ] || [ -z "$BASENAME" ]; then
    echo "Usage: upload <log_file_path> <log_file_name>"
    exit 1
fi

if [ ! -f "$LOG_PATH" ]; then
    echo "❌ Error: File $LOG_PATH not found."
    exit 1
fi

# Timestamped filename to avoid collisions
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
FILENAME="${BASENAME}_${TIMESTAMP}.log"
FILE_PATH="$FILENAME"

# -------------- ENCODE FILE CONTENT --------------
ENCODED_CONTENT=$(base64 -w 0 "$LOG_PATH")  # -w 0 disables line-wrapping
# -----------------------------------------------

# -------------- CREATE TEMP JSON PAYLOAD --------------
PAYLOAD=$(mktemp)
cat <<EOF > "$PAYLOAD"
{
  "message": "Upload $FILENAME",
  "branch": "$BRANCH",
  "content": "$ENCODED_CONTENT"
}
EOF
# --------------------------------------------------------

# -------------- API CALL USING JSON PAYLOAD --------------
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  /repos/$GITHUB_USER/$REPO/contents/$FILE_PATH \
  --input "$PAYLOAD"
# ---------------------------------------------------------

# -------------- OUTPUT LINK --------------
echo "✅ Uploaded: https://github.com/$GITHUB_USER/$REPO/blob/$BRANCH/$FILE_PATH"

# -------------- CLEANUP --------------
rm "$PAYLOAD"
