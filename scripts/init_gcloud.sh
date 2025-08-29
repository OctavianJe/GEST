set -euo pipefail

file=$(ls .secrets/gcloud/client_secret*.json 2>/dev/null | head -n1) || true
if [[ -z "$file" ]]; then
  echo "No client_secret*.json" >&2
  exit 1
fi

exec gcloud auth application-default login \
    --client-id-file="$file" \
    --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/gmail.send \
    --no-browser