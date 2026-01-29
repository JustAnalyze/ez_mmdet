# Product Guidelines: ez_mmdet

## Tone & Voice

- **Helpful & Conversational:** We use a friendly, approachable tone. Documentation and terminal messages should feel like a guided experience rather than a formal technical manual.
- **Partnership:** We use "We" and "Us" to create a sense of collaboration with the user (e.g., "We found an issue with your dataset configuration...").
- **Encouraging:** Use active, positive verbs like "Launch," "Start," and "Succeed" to motivate users.

## Visual Identity & Output

- **Rich Terminal Experience:** We leverage the `rich` library to provide beautiful, color-coded logs, progress bars, and formatted tables for results.
- **Clarity over Density:** Avoid overwhelming the user with wall-of-text logs. Use color and spacing to highlight important events and hide unnecessary boilerplate.

## User Interface (API & CLI)

- **Progressive Disclosure:** Our primary interfaces are designed for absolute simplicity by default. We use sensible defaults so users can start training with minimal input, but we allow advanced users to "unlock" deeper MMDetection customization through optional pass-through parameters.
- **Intuitive Naming:** We prefer descriptive, user-centric names over internal framework jargon.

## Error Handling & Troubleshooting

- **Solution-Oriented:** Every error message should provide a clear reason for the failure and, whenever possible, an actionable step to fix it (e.g., "We couldn't find your annotations. Please verify the `ann_file` path in your `dataset.toml`.").
- **Early Validation:** Use Pydantic to catch configuration errors before the underlying engine starts, delivering clean and human-readable validation reports.
