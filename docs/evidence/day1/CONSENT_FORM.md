# CONSENT_FORM.md
## Participant Information Sheet and Consent Form
### AAA Engine — Voice Recording for Speaker Verification Evaluation

**Version:** 1.0 | **Date:** 2026-11-11 | **Project:** AAA Engine Sprint, Nov 2026

---

## Part A: Participant Information Sheet

### What is this study?

We are building an Adaptive Acoustic Authentication Engine (AAA Engine) that uses voice to verify speaker identity. As part of this project, we need voice recordings from real participants for two purposes:

- **Speaker verification evaluation** — to test how well our system can confirm that a person is who they claim to be.
- **Automatic speech recognition evaluation** (optional — see scope options below).

### Who is running this study?

This study is being conducted by the AAA Engine development team. Contact: [CONTACT EMAIL: PENDING ETHICS / SPONSOR CONFIRMATION].

### What will happen?

If you agree to participate, you will be asked to:
1. Read a set of challenge phrases aloud (provided on a recording device or screen).
2. Speak freely about a neutral topic for 1–2 minutes.
3. Optionally attend a second session at least one day later, using a different device.

Each session takes approximately **15–20 minutes**.

### What data is collected?

- Audio recordings of your voice.
- Device type used for recording (e.g., phone model, laptop model).
- Session date and approximate recording conditions (quiet / noisy).
- A pseudonymous **speaker ID** (a random code, not your name).

**Your name, email address, and other identifying information are NOT recorded in the audio files or the dataset.** The link between your name and your speaker ID is kept in a separate key table, stored securely and accessed only by the principal investigators.

### What are the scope options?

Please tick the purposes for which you consent to your recordings being used:

| Scope code | Meaning |
|---|---|
| `sv_enroll` | Speaker verification — enrolment data (reference voice) |
| `sv_eval` | Speaker verification — evaluation data (test trials) |
| `asr_train` | Automatic speech recognition — model training |
| `eval_only` | Evaluation only (not training) |

You may select any combination. If you select none, your data will not be used.

### How long will data be kept?

Audio recordings will be kept for [RETENTION PERIOD: PENDING ETHICS / LEGAL APPROVAL] (proposed: 24 months from the date of recording, then securely deleted). You may request deletion at any time (see below).

### Who will have access?

[ACCESS TERMS: PENDING ETHICS / LEGAL APPROVAL] Access to audio files is restricted to authorized members of the AAA Engine development team. Data will not be shared with third parties without your explicit additional consent. No recordings will be published or made publicly accessible.

### Can I withdraw?

Yes. You may withdraw at any time before [WITHDRAWAL DEADLINE: PENDING ETHICS / LEGAL APPROVAL] (proposed: Day 3 test-data freeze, 2026-11-13 18:00 IST) without giving a reason. After the freeze, anonymised recordings that have been included in published evaluation results cannot be removed from those results, but will be deleted from storage.

To withdraw or request data deletion, contact: [CONTACT EMAIL: PENDING ETHICS / SPONSOR CONFIRMATION] with your **consent ID** (provided at the bottom of your signed form). Do not send your name in the email subject line.

---

## Part B: Consent Form

**Please read carefully and sign/initial each item.**

**Consent ID:** `CONSENT-YYYYMMDD-NNNN` *(filled in by operator — see consent_id scheme below)*

| # | Statement | I agree (✓) |
|---|---|---|
| 1 | I have read and understood the Participant Information Sheet above. | |
| 2 | I understand that my voice will be recorded and used for research as described. | |
| 3 | I understand that my recordings will be stored pseudonymously (speaker ID, not my name). | |
| 4 | I understand that only the AAA Engine team will have access to my recordings. | |
| 5 | I understand that I may withdraw before the Day 3 test-data freeze. | |
| 6 | I confirm that I am at least 18 years old. | |

**Scope options** (tick all that apply):

- [ ] `sv_enroll` — Speaker verification enrolment
- [ ] `sv_eval` — Speaker verification evaluation
- [ ] `asr_train` — ASR model training
- [ ] `eval_only` — Evaluation purposes only

**Participant pseudonym assigned by operator:** `SPK-YYYYMMDD-NNNN` *(do not write your real name)*

**Date:** _______________

**Participant signature:** _______________

**Operator initials:** _______________

---

## Consent ID Scheme

Each participant receives a unique `consent_id` at the time of signing:

```
Format: CONSENT-{YYYYMMDD}-{4-digit sequence}
Example: CONSENT-20261112-0003
```

The sequence resets daily. The operator assigns the next available ID from a central log (maintained in a secure, non-Git spreadsheet). The consent ID is written on the physical/digital form and also recorded in the metadata CSV for the recording session.

**The consent_id → real name mapping is stored only in the key table (never in audio filenames, manifests, or Git).**

---

## Operator Notes

- Do NOT write participants' real names in any filename or digital record.
- Collect the signed form (physical or PDF) before recording begins.
- Record the `consent_id` and scope selections in the metadata CSV immediately after signing.
- If a participant withdraws before the freeze, delete their audio files and remove their rows from the manifest. Log the deletion in the exclusion report.
- Store signed consent forms in a locked folder / encrypted drive separate from the audio data.
