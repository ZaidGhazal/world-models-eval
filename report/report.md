# DreamGrasp: Can You Trust a Robot Test That Happens Entirely Inside an AI's Imagination?

## Terms used in this report

Every technical term in this report, defined once here in plain language. Each is also re-explained where it first appears.

- **Policy**: the robot's control program. It looks at a camera image and a written instruction, and decides how to move the arm. When we say "the robot," we usually mean this program.
- **Checkpoint (or snapshot)**: a saved copy of the policy at a specific point during its training. An early checkpoint is clumsy; a late one is skilled. We saved 8 of them.
- **World model (or "imagination")**: a second, separate AI trained to predict what the robot's camera would see next, given the current view and the robot's chosen movement. It replaces the physics simulator during a dream.
- **Tier**: one of our five world models. We deliberately built them at different quality levels (tier 1 through tier 5) so we could study how quality affects trustworthiness.
- **Dream (or dream rollout)**: one imagined test run. The policy and the world model pass images and actions back and forth for a set number of steps, producing a short imagined video of the robot attempting a task. No simulator or hardware is involved.
- **Rollout**: one test run in general, whether real (in the simulator) or imagined (a dream).
- **Simulator**: the physics engine (LIBERO, built on MuJoCo) that actually simulates the arm, the objects, gravity, and contact. Its results are our ground truth.
- **Ground truth**: the trusted correct answer we compare everything against. Here, it is 12,000 real simulator runs, where success is checked physically (is the bowl actually on the plate?).
- **The judge (success classifier)**: a third AI that watches a video of a rollout and outputs a score for "this looks like a success." We trained it on real simulator videos with known outcomes. It is needed because inside a dream there is no physics engine to check the outcome.
- **Held-out tasks**: tasks we deliberately excluded from the robot's training, used to test how everything behaves when the robot faces something unfamiliar.
- **Distribution shift**: the situation where a model faces inputs unlike its training data. Our held-out tasks create it on purpose.
- **Horizon (T)**: how many steps a dream runs. Our main setting is T=200 steps; one stress test shortens it to T=100.
- **N (number of dreams)**: how many separate dreams we run per checkpoint per tier before averaging. Main setting: 50. One stress test drops it to 20.
- **PSNR / SSIM / LPIPS**: three standard scores for how similar two images are. PSNR and SSIM: higher is better. LPIPS: lower is better. We use them to measure how closely a dreamed frame matches the real frame.
- **Divergence step**: how many steps a dream runs, on average, before its prediction errors snowball past a fixed threshold. Plainly: how long the imagination stays believable.
- **Spearman correlation (agreement score)**: a number from -1.0 to 1.0 measuring how well two rankings agree. 1.0 means identical rankings, 0 means no relationship, -1.0 means exactly backwards. We use it to compare each tier's ranking of the 8 checkpoints against the simulator's ranking.
- **Pearson correlation**: a similar -1.0 to 1.0 number, but for how two sets of scores move together (rather than rankings). We use it to compare dream scores against real success rates, task by task.
- **Confidence interval (CI)**: a statistical range around a measured number expressing how much it could move just from luck. A wide CI means the number is not precisely pinned down. A CI that crosses zero means we cannot even be confident of the number's sign.
- **Bootstrap**: the resampling technique we use to compute those confidence intervals: reshuffle the existing results many times and see how much the number wobbles.
- **LPIPS loss (tier 5 only)**: an extra training signal that rewards visually sharp predicted frames. Meant to make tier 5 the best; section 3.3 shows what actually happened.
- **Seed**: a fixed starting number for randomness, used so runs can be repeated. (Limitation 8 explains why our dreams still are not exactly repeatable despite fixed seeds.)

## Abstract 

Testing a robot is expensive. You need hardware, time, and a safe place to let it fail. So researchers had an idea: what if we test the robot inside an AI's imagination instead? Train a second AI, called a world model, to predict what the robot's camera would see next. Then let the robot practice inside that prediction, thousands of times, for almost free.

Recent papers show this can work. But they all used one world model, of one quality level. Nobody has asked the obvious follow-up: how good does the imagination have to be before you can believe its test results? A bad imagination that gives confident wrong answers is worse than no test at all.

We built an experiment to measure exactly that. We trained five imaginations, on purpose making some better and some worse. We took one robot control program at eight different stages of its training, from clumsy to skilled. Then we tested every stage inside every imagination, and compared those imaginary test scores against 12,000 tests in a real physics simulator, which served as our answer key.

Two findings held up no matter how we stress-tested them:

1. **The imaginary test fails silently on unfamiliar tasks.** On tasks the robot never trained on, it scored exactly 0% in the real simulator. Total failure, every time. But the imaginary test never noticed. It kept handing out normal-looking, confident scores. Nothing in the imaginary results warned us that the robot was completely lost.
2. **The judge grades the wrong thing.** Our automated judge (an AI that watches the imagined video and decides "success or failure") seems to partly grade how clean the video looks, not whether the robot finished the job. Our proof: when we made the imagined videos shorter, so the imagination had less time to get blurry and glitchy, scores went up for all five imaginations. If the judge were truly grading success, shorter tests should have scored lower, because most tasks need more time than that to finish.

Because of these problems, our attempt to rank the five imaginations from most to least trustworthy did not survive stress-testing: change small details of the test and the ranking reshuffles. Eight robot stages is too few for the statistics to settle it. So we present our ranking as a demonstration of the method, not a final verdict. All of our code, data, and models are released openly, and everything runs on one ordinary GPU.

## 1. Introduction

Here is the problem in one sentence: checking whether a robot control program is any good requires running it many times, and every way of doing that is costly. Real hardware is slow and can break things. Physics simulators are cheaper, but still take hours for thorough testing.

So a new line of research proposes a third option. Train a world model, an AI whose only job is to answer "if the robot does X, what will the camera see next?" Then run the robot inside that AI's predictions. No physics, no hardware, just the robot's controller and the world model passing images and actions back and forth. Researchers call one of these imagined test runs a "dream."

Several groups have shown promising versions of this: WorldEval, WPE, Ctrl-World, SIMPLER, and RoboWM-Bench (full citations at the end). We did not invent this idea, and we want to be clear about that. What none of those papers did is calibrate it. Each used one world model at one quality level. None measured the question a practitioner actually faces: how good does my world model need to be before I can act on its verdicts?

That is what this study measures. Our two contributions:

1. We found two specific ways dream-based testing misleads you: it gives no warning when the robot faces unfamiliar tasks, and its automated judge partly grades video quality instead of task success.
2. We built and released the full experiment as open code that runs on a single GPU, together with an honest analysis showing that our own ranking of the five world models is not statistically solid, and what it would take to make it solid.

## 2. Method: what we built and how we tested it

### 2.1 The robot and the answer key

The robot tasks come from LIBERO, a standard benchmark of tabletop jobs like "put the bowl on the plate," simulated with a robot arm. We trained a control program (SmolVLA, a model that takes a camera image plus a written instruction and outputs arm movements) and saved a snapshot of it every 5,000 training steps, giving us 8 snapshots from early-and-clumsy to fully-trained. On the task suite this study focuses on, the worst snapshot succeeds about 2% of the time and the best about 29%. That 27-point range matters: to test whether the imaginary test can tell good from bad, you need snapshots that genuinely are good and bad.

The answer key: we ran every snapshot 50 times on every task in the real physics simulator. 12,000 test runs in total. The simulator physically checks the outcome (is the bowl actually on the plate?), so these results are ground truth.

### 2.2 The five imaginations

Each world model has two parts: a compressor that squeezes each camera image down to a small grid of numbers, and a predictor that learns "given the recent images and the robot's action, what comes next?"

We trained five versions, deliberately varied (Table 1): tier 1 saw only 10% of the training data, tier 2 saw 25%, tier 3 saw 50% with a bigger brain, tier 4 saw everything, and tier 5 saw everything plus two extras meant to make it the best: a longer memory of past frames, and an added training signal (LPIPS) that rewards visually sharp predictions. Everything else was kept identical across the five, so any difference in results comes from these design choices alone.

**Table 1. World-model tier design.**

| Tier | Data fraction | Context (frames of memory) | Predictor size | LPIPS weight |
|---|---:|---:|---|---:|
| tier_1 | 0.10 | 2 | 6 layers / 256 wide | 0.0 |
| tier_2 | 0.25 | 2 | 6 layers / 256 wide | 0.0 |
| tier_3 | 0.50 | 4 | 12 layers / 512 wide | 0.0 |
| tier_4 | 1.00 | 4 | 12 layers / 512 wide | 0.0 |
| tier_5 | 1.00 | 8 | 12 layers / 512 wide | 0.1 |

### 2.3 How a dream test works

Start from a real camera frame. The robot's controller looks at it and picks an action. Instead of a simulator executing that action, the world model predicts the next frame. The controller looks at the predicted frame and picks its next action. Repeat for 200 steps, and you get a short imagined video of the robot attempting the task, with no simulator involved.

Then a judge scores it. The judge is a separate AI we trained on thousands of real simulator videos that came with true success/failure labels. On videos it had never seen, the judge agreed with the true label 98.17% of the time. That sounds excellent, but hold onto a caveat: 98% accurate on real videos does not guarantee it judges imagined videos the same way. That caveat becomes the story of section 3.2.

### 2.4 How we measure things

Two kinds of measurement:

**How good is each imagination at predicting?** We compare predicted frames against real ones using three standard image-similarity scores (PSNR, SSIM, LPIPS; the first two are better when higher, the third when lower), at 1, 8, 16, and 32 steps into the future. We also record the "divergence step": how many steps a dream typically runs before its errors snowball past a fixed threshold. Think of it as how long the imagination stays believable.

**How trustworthy is each imagination as a tester?** We let each imagination rank the 8 robot snapshots from worst to best, and compare that ranking to the real simulator's ranking. The agreement score (Spearman correlation) runs from 1.0 (identical rankings) through 0 (no relationship) to -1.0 (exactly backwards). We wrap every such number in a confidence interval, a statistical range expressing how much the number could move just from luck.

### 2.5 Stress tests, planned in advance

Before running anything, our project plan committed us to re-checking the results under altered conditions, so a fluke would not survive. The three stress tests: use 20 dreams per test instead of 50; nudge the judge's pass/fail cutoff up and down; and rerun everything with dreams of 100 steps instead of 200. The first two are free reanalysis of data we already had. The third required a fresh set of dreams.

## 3. Results

### 3.1 The imaginary test fails silently on unfamiliar tasks

We deliberately held two tasks out of the robot's training (held-out tasks: tasks the policy has never seen, used to test behavior on the unfamiliar), to see what happens when it faces something new. The real simulator's answer was brutal and unambiguous: 0 successes in 800 attempts, across all 8 snapshots. The robot cannot do these tasks at all.

Now the important part. When we ran those same tasks as dreams, the imaginary test did not go quiet or throw a warning. It produced the same kind of confident, nicely spread-out scores it produces everywhere else; one imagination scored the snapshots anywhere from 0.26 to 0.51, as if some were decent and others mediocre at a task none of them can do whatsoever.

Imagine relying on this in practice, with no real-world check: you would see reasonable-looking test scores and have no idea your robot is completely lost. That is the study's central warning. When the robot leaves familiar territory, the dream test does not degrade visibly. It fails silently.

### 3.2 The judge grades how the video looks, not just what happened in it

This finding started with a number that made no sense. We ran a stress test with shorter dreams: 100 steps instead of 200. Since the average successful task takes about 104 steps to complete, shorter dreams should mean fewer visible successes and lower scores.

Scores went up instead. For every single imagination. The average score rose from 0.185 to 0.236.

Why would cutting the test short raise scores? Here is the mechanism we believe explains it. Each imagination predicts frame after frame, feeding on its own output, so small errors snowball: every one of the five gets measurably blurrier and more distorted the longer a dream runs (section 3.3 has the numbers). A 100-step dream simply ends before much of that decay sets in, so the video looks cleaner. If the judge's score goes up whenever the video looks cleaner, regardless of whether the task got done, then the judge is at least partly grading video quality, not task success.

Two more observations point the same way. First, across tasks, the judge's scores and real success rates are negatively related in every condition we tested (correlation of -0.492, -0.297, and -0.452 across the three settings; a negative number here means that on tasks where the robot truly does better, the judge tends to score its dreams *lower*, not higher). Tasks the robot is genuinely better at do not get better dream scores; if anything the opposite. Second, the single strangest result in the study: the robot's best snapshot by real testing (29% real success) was ranked in the middle by 200-step dreams and dead last by 100-step dreams. The length of the imagined video changed the verdict on the same robot.

Honesty requires a caveat: we have not tested this mechanism directly. Everything above is consistent with a judge that grades looks, but we did not isolate it from all alternatives. Section 3.5 spells out the experiment that would settle it: hand the judge matched pairs of clean-looking failures and messy-looking successes, and see which it prefers.

### 3.3 How good each imagination actually is

**Table 2. Prediction quality by tier and horizon (PSNR / SSIM / LPIPS), and average divergence step.**

| Tier | 1 step ahead | 8 steps | 16 steps | 32 steps | Divergence step |
|---|---|---|---|---|---:|
| tier_1 | 26.81 / .943 / .134 | 21.08 / .748 / .250 | 19.47 / .692 / .294 | 17.97 / .661 / .320 | 19.81 |
| tier_2 | 27.04 / .939 / .132 | 21.63 / .752 / .245 | 20.28 / .721 / .272 | 18.65 / .698 / .306 | 20.94 |
| tier_3 | 30.28 / .984 / .046 | 26.42 / .955 / .079 | 24.95 / .938 / .096 | 22.54 / .900 / .121 | 20.38 |
| tier_4 | 31.98 / .990 / .030 | 29.32 / .977 / .040 | 27.49 / .965 / .045 | 25.00 / .946 / .056 | 23.88 |
| tier_5 | 31.49 / .989 / .028 | 26.44 / .964 / .077 | 23.84 / .935 / .121 | 21.19 / .883 / .184 | 12.94 |

Two patterns to take from this table.

First, read across any row: every imagination gets worse the further ahead it predicts. No exceptions. This steady decay is what makes the section 3.2 argument work.

Second, read down the divergence-step column, and notice the surprise at the bottom. Tier 4 is the best imagination in the family on every measure. But tier 5, which was designed to be the best (most data, longest memory, extra sharpness training), stays believable for the fewest steps of all five: about 13, versus 20 to 24 for the others. We checked this by eye rather than trusting the numbers: put tier 5's dreams next to the real footage and they track faithfully for the first dozen steps, then the arm and objects visibly dissolve into fragments. This is a real collapse, not a scoring glitch.

Our best explanation, not yet confirmed: the extra sharpness training rewards each single frame for looking crisp, even if that pulls the model away from staying accurate over many frames, and the longer memory means the model conditions on more of its own accumulating mistakes. More sophistication made it worse. We kept tier 5 in every analysis rather than quietly dropping it; a below-expectations imagination is exactly the kind of thing this calibration study exists to measure.

### 3.4 The trustworthiness ranking, illustrated, not settled

**Read this section as a demo of the method, not a final ranking. The next section shows why.**

At our main test settings, the picture looked striking, even exciting: the two simplest, weakest imaginations were the most trustworthy testers (agreement scores of 0.881 and 0.810 with the real ranking, where 1.0 would mean their ranking of the 8 snapshots matched the simulator's ranking exactly), and trustworthiness fell as the imaginations got fancier (0.595, 0.548, 0.524). Better-looking predictions did not mean better test verdicts anywhere in the family.

Then the stress tests took it apart. Using 20 dreams instead of 50 flipped tier 4's score from mildly positive to negative. Shortening dreams to 100 steps collapsed the two frontrunners to roughly zero and made mid-pack tier 3 the new leader. With only five imaginations and eight robot snapshots, these agreement scores swing too easily for any single version of the ranking to be trusted.

### 3.5 The stress tests, in full

**Table 3. Ranking-agreement score for each imagination, under every condition tested.**

| Tier | Main setting (50 dreams, 200 steps) | Judge cutoff 0.4 | Judge cutoff 0.6 | 20 dreams | 100 steps |
|---|---:|---:|---:|---:|---:|
| tier_1 | 0.881 | 0.857 | 0.786 | 0.857 | 0.024 |
| tier_2 | 0.810 | 0.731 | 0.874 | 0.690 | 0.071 |
| tier_3 | 0.595 | 0.405 | 0.619 | 0.405 | 0.571 |
| tier_4 | 0.548 | 0.756 | 0.577 | **-0.143** | 0.190 |
| tier_5 | 0.524 | 0.452 | 0.429 | 0.357 | 0.238 |

How to read it: each column changes one thing about the test; a trustworthy result should look similar across a row. The judge-cutoff columns do stay similar, so the ranking is not an artifact of where the pass/fail line sits. The 20-dream column mostly holds but flips tier 4 to negative. The 100-step column is the decisive break: the two leaders drop to roughly zero.

Before accepting that as real, we tried to explain it away twice, and failed both times. Explanation one: maybe 100 steps just cuts tasks off before they can finish. But that would lower scores, and scores rose (section 3.2). Ruled out. Explanation two: maybe one weird snapshot is dragging everything. We removed each snapshot one at a time and recomputed; the best case recovered only half the gap. And a closer look showed the 200-step ranking was already fuzzy only among the four nearly-tied top snapshots (real success rates within 6 points of each other, close enough to be luck), while the 100-step collapse went further, flipping individual snapshots outright. So: partly, but not mostly, a fluke of particular snapshots.

One measurement did stay stable through every condition: the task-by-task comparison from section 3.2, which was consistently negative (-0.492, -0.297, -0.452). Stable, and pointing at the judge problem.

Bottom line: this study cannot deliver a confident ranking of five imaginations from eight robot snapshots. Getting one would take either many more snapshots and repeated dreams (more statistical power), or fixing the judge first, via the direct audit described above: matched pairs of clean-looking failures and messy-looking successes.

## 4. Limitations

The full list lives in LIMITATIONS.md. The short version: everything here happens in simulation, with small models, one robot arm type, and one benchmark's tasks, and every dream score inherits the judge's imperfections as a ceiling. Three limitations do real work in this report rather than being boilerplate: item 7 is the held-out silent-failure result behind section 3.1; item 8 documents that dreams are not exactly repeatable run-to-run even with a fixed random seed (tiny GPU arithmetic differences snowball over 200 steps), which is why the 20-dream stress test re-uses existing dreams instead of generating new ones; and item 9 is the too-few-snapshots problem that forces section 3.4 to be read as an illustration.

## 5. Reproducibility

`bash scripts/run_study.sh` regenerates the main chart and numbers from `results/sim_success.parquet` and `results/dream_success.parquet`. The 100-step stress test is its own artifact, `results/dream_success_t100.parquet`, produced by `scripts/run_t26_horizon100.sh` and analyzed with `python -m dreamgrasp.eval.correlate --dream results/dream_success_t100.parquet`. The judge-cutoff and 20-dream checks recompute from data already on disk via `--threshold-sweep 0.4 0.6` and `--n-dreams-cap 20`.

## References

- [worldeval2025] WorldEval: World Model as Real-World Robot Policies Evaluator. arXiv:2505.19017
- [wpe2025] World-model-based Policy Evaluation. arXiv:2506.00613
- [ctrlworld2025] Ctrl-World: A Controllable Generative World Model for Robot Manipulation. arXiv:2510.10125
- [simpler2024] Evaluating Real-World Robot Manipulation Policies in Simulation. arXiv:2405.05941
- [robowmbench2026] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. arXiv:2604.19092
- [libero2023] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310
- [smolvla2025] SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844
