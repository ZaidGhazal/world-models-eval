# Release Approval Checklist

T2.7 stops before public release actions. Request these approvals separately; do not bundle them.

## Approval Items

- VLA checkpoint upload to Hugging Face: pending user approval.
- World-model checkpoint upload to Hugging Face: pending user approval.
- Gradio Space publication: pending user approval.
- GitHub `v1.0` tag: pending user approval.

## Verification After Approval

For each approved item, record:

- Artifact URL
- Commit or HF revision
- Cross-links checked from README, HF card, and Space
- Smoke check result

Do not mark T2.8 complete until all approved artifacts are live and `scripts/reproduce.sh`
regenerates the trust-region chart from raw parquets.
