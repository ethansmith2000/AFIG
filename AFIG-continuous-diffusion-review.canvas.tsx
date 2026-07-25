import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Tab = "design" | "representation" | "training" | "experiments";

const architectureRows = [
  ["Input token", "`xᵢ ∈ R⁶`: RGB real/imaginary coefficient at one frequency", "Avoid polar discontinuities"],
  ["Backbone", "Causal Transformer over shifted continuous tokens", "Produces `zᵢ = f(x<ᵢ)`"],
  ["Output model", "3-block AdaLN residual MLP, width 256–512", "Models `p(xᵢ | zᵢ)`"],
  ["Training diffusion", "1,000-step cosine schedule, ε-prediction", "Closest controlled reproduction of MAR"],
  ["Inference", "DDIM, 10–25 steps initially, `clip_sample=False`", "Low-dimensional head should not assume 100 steps"],
  ["Objective multiplier", "4 independent `(t, ε)` pairs per `(xᵢ, zᵢ)`", "Reduces Monte Carlo variance; not four new contexts"],
];

const representationRows = [
  [
    "Real + imaginary",
    "6",
    "Linear, no phase wrap; ordinary Gaussian diffusion fits naturally; exact complex value",
    "Frequency-dependent scale; requires careful normalization",
    "Recommended baseline",
  ],
  [
    "Log magnitude + raw phase",
    "6",
    "Interpretable and close to the current code",
    "Discontinuity at ±π; phase is meaningless near zero magnitude",
    "Do not use with plain MSE diffusion",
  ],
  [
    "Log magnitude + sin/cos phase",
    "9",
    "Circularly continuous",
    "Off-manifold diffusion samples; must renormalize pairs; undefined phase remains",
    "Useful polar ablation",
  ],
  [
    "Special circular diffusion",
    "6",
    "Respects angular geometry directly",
    "Custom scheduler/process; defeats the simple Diffusers implementation",
    "Later research path",
  ],
];

const normalizationRows = [
  [
    "Global 6-vector stats",
    "Simple; preserves relative spectral energy",
    "Low frequencies dominate scale and high frequencies live near zero",
    "Sanity baseline only",
  ],
  [
    "Per-position × component",
    "Every diffusion coordinate is near unit variance",
    "Strongly reweights high frequencies; 3,264 fitted scalar stats",
    "Diagnostic upper bound",
  ],
  [
    "Radial-bin × RGB × real/imag",
    "Stable estimates; follows spectral scale; transferable across nearby positions",
    "Still changes the implicit image-space weighting",
    "Recommended first model",
  ],
  [
    "6×6 whitening per radial bin",
    "Makes isotropic noise meaningful and removes RGB/component covariance",
    "Can hide useful structure from the head; inversion must be exact",
    "Second-stage ablation",
  ],
];

const exposureRows = [
  [
    "Sequence-level strength + tokenwise white noise",
    "Sample one `σ` per image; draw independent normalized noise per prior token",
    "Coherent corruption severity with cheap parallel training",
    "Start here: 50% clean, otherwise small log-uniform σ",
  ],
  [
    "Independent strength per token",
    "Each history token gets unrelated severity",
    "Produces salt-and-pepper histories unlike normal rollouts",
    "Avoid as the default",
  ],
  [
    "Colored noise in raw Fourier space",
    "Match empirical ring/channel covariance",
    "Principled only before whitening; easy to double-count covariance",
    "Use only if normalized rollout residuals are clearly colored",
  ],
  [
    "Model-sampled replacement",
    "Replace a prefix subset with stop-gradient model samples in a second pass",
    "Closest to inference errors, but expensive and nonstationary",
    "Add only after measuring actual rollout residuals",
  ],
  [
    "Token masking",
    "Replace some history tokens and add an explicit missingness embedding",
    "Robust to missing context, not necessarily small numerical drift",
    "Complement, not substitute, for perturbation",
  ],
];

const experimentRows = [
  ["0", "Transform round-trip", "`max|x − inverse(forward(x))|`, imaginary residual, constraint violations", "Must pass before training"],
  ["1", "One-step density baselines", "Full-covariance Gaussian and small conditional GMM/MDN", "Test whether diffusion is actually needed for 6D"],
  ["2", "DiffLoss smoke", "Cartesian, radial normalization, multiplier 1, DDIM 20", "Validate loss decrease and samples"],
  ["3", "Multiplier", "`N ∈ {1, 4}` at matched wall-clock and optimizer steps", "Measure value of reused conditions"],
  ["4", "Sampler", "DDIM steps `{5, 10, 20, 50}`", "Find the real latency-quality frontier"],
  ["5", "Robustness", "Clean teacher forcing vs sequence-level input noise", "Measure late-ring drift"],
  ["6", "Representation", "Cartesian vs logmag + unit-vector phase", "Only after both have exact transforms"],
  ["7", "Sequence granularity", "Coefficient tokens vs chunks/rings or masked generation", "Largest possible speed win"],
];

function Design() {
  return (
    <Stack gap={18}>
      <Callout tone="success" title="Recommendation">
        Build the first continuous version with six standardized Cartesian FFT components per
        frequency, a causal Transformer, and a MAR-style AdaLN denoising MLP. This cuts the current
        3,264 scalar AR steps to 544 coefficient steps immediately. A canonical unique-spectrum
        layout can reduce that further to 514.
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={16}>
        <Stack gap={8}>
          <H2>Minimal architecture</H2>
          <Table
            headers={["Piece", "Choice", "Reason"]}
            rows={architectureRows}
            rowTone={["success", "neutral", "neutral", "info", "info", "neutral"]}
            striped
          />
        </Stack>

        <Card size="lg">
          <CardHeader trailing={<Pill active>Key flow</Pill>}>Training shapes</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>`tokens: [B, L, 6]`</Text>
              <Text>`z = transformer(BOS, tokens[:, :-1]): [B, L, D]`</Text>
              <Text>`x₀ = tokens: [B, L, 6]`</Text>
              <Divider />
              <Text>`z, x₀ → flatten to [B·L, ...]`</Text>
              <Text>`repeat N times → [N·B·L, ...]`</Text>
              <Text>`t, ε sampled independently per repeated row`</Text>
              <Text tone="secondary">
                Average over all rows so increasing N does not multiply the gradient scale.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="warning" title="Diffusion may be overkill in only six dimensions">
        MAR used diffusion to model continuous latent tokens with richer distributions. A Fourier
        coefficient conditioned on a long low-frequency prefix—especially at high frequencies—may
        be close to a correlated Gaussian. A full-covariance Gaussian or a small mixture-density
        head samples once, versus 544 × 10–25 denoiser calls. Treat this as an empirical question,
        not a foregone conclusion.
      </Callout>

      <Stack gap={8}>
        <H2>The larger speed opportunity</H2>
        <Text>
          Six-dimensional coefficient tokens give a 6× AR-step reduction, but sampling still nests
          a diffusion loop inside every coefficient step. The next experiment should be fixed-size
          chunks, entire frequency shells, or a frequency-constrained masked generator that predicts
          several same-ring coefficients together. Later work independently moved this way:
          <Text as="span"> </Text>
          <Link href="https://arxiv.org/abs/2503.05305">FAR</Link>
          <Text as="span"> and </Text>
          <Link href="https://arxiv.org/abs/2503.07076">NFIG</Link>.
        </Text>
        <Text tone="secondary">
          Your current traversal is a square spiral (L∞ shells), not exact Euclidean Fourier rings.
          Sorting or grouping by normalized radius would make the causal claim and ring-level
          experiments cleaner.
        </Text>
      </Stack>
    </Stack>
  );
}

function Representation() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Choose coordinates before choosing the loss</H2>
        <Table
          headers={["Representation", "Dims", "Strength", "Failure mode", "Verdict"]}
          rows={representationRows}
          rowTone={["success", "danger", "warning", "neutral"]}
          striped
        />
      </Stack>

      <Callout tone="info" title="Why Cartesian is the clean baseline">
        `Re(F)` and `Im(F)` contain exactly the same information as magnitude and phase, but they
        remove the branch cut at ±π and naturally downweight phase when magnitude is near zero.
        This is precisely where raw phase is statistically unstable. Use `torch.fft` with
        `norm="ortho"` so scaling is resolution-independent and Parseval reasoning remains clean.
      </Callout>

      <Stack gap={8}>
        <H2>Normalization is also a loss-weighting decision</H2>
        <Table
          headers={["Scheme", "Benefit", "Cost", "Use"]}
          rows={normalizationRows}
          rowTone={["neutral", "warning", "success", "info"]}
          striped
        />
        <Text tone="secondary">
          Do not normalize per image unless the removed mean/scale is separately generated or
          transmitted. Dataset-level statistics are invertible at generation time; per-image
          statistics silently discard brightness and contrast information.
        </Text>
      </Stack>

      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader>Hermitian validity</CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>A real 32×32 image has 1,024 real Fourier degrees of freedom per channel.</Text>
              <Text>
                The current 32×17 half-plane stores 544 complex locations, so boundary frequencies
                include redundant conjugate constraints.
              </Text>
              <Text tone="secondary">
                Either generate a 514-location canonical set, or explicitly project/enforce
                boundary conjugacy before feedback and reconstruction.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Special frequencies</CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>DC and Nyquist self-conjugate points must be real.</Text>
              <Text>Boundary conjugate pairs must be exact, not merely encouraged by loss.</Text>
              <Text tone="secondary">
                Prefer deterministic parameterization or component masks over asking diffusion to
                learn a measure-zero algebraic constraint.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function Training() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Scheduler and denoiser</H2>
        <Text>
          To match the paper first, use a 1,000-step cosine training schedule and ε-prediction.
          For Diffusers, keep training and DDIM inference beta configurations identical, call
          `set_timesteps`, scale model input if the scheduler requires it, and disable sample
          clipping because normalized Fourier values are unbounded.
        </Text>
        <Text tone="secondary">
          Do not import Stable Diffusion’s zero-terminal-SNR and guidance-rescale defaults without
          evidence. Those recommendations target image-latent brightness behavior. `v_prediction`
          is a useful ablation, not a prerequisite here.
        </Text>
      </Stack>

      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader>MLP block</CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>`h = Linear(x_t)`</Text>
              <Text>`c = TimeEmbed(t) + Linear(z)`</Text>
              <Text>`h += gate(c) · MLP(AdaLN(h, c))` × 3</Text>
              <Text>`ε̂ = FinalAdaLNLinear(h, c)`</Text>
              <Text tone="secondary">
                Zero-initialize modulation and final output as in DiT/MAR. Output 6 values; learned
                variance is unnecessary for a DDIM-first implementation.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Condition dropout / CFG</CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>For unconditional CIFAR-10, CFG is not automatically useful.</Text>
              <Text>
                Dropping `z` asks the head to model `p(xᵢ)` instead of `p(xᵢ|x&lt;ᵢ)` and can amplify
                history dependence rather than class adherence.
              </Text>
              <Text tone="secondary">
                Start with no CFG. If class-conditioning is added, drop the class condition in the
                Transformer, preserve position, and use conventional class CFG.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>Exposure-bias strategy</H2>
        <Table
          headers={["Method", "Granularity", "Interpretation", "Recommendation"]}
          rows={exposureRows}
          rowTone={["success", "warning", "neutral", "info", "neutral"]}
          striped
        />
      </Stack>

      <Callout tone="warning" title="Measure rollout errors before designing colored noise">
        Whitened token space makes isotropic perturbations a defensible first approximation. After a
        baseline trains, run free generations and compare generated-prefix errors with teacher-forced
        residuals by radius, RGB/component covariance, and prefix length. Fit the corruption process
        to that evidence. Otherwise “colored” noise can encode the data covariance twice and still
        miss the non-Gaussian, state-dependent errors that matter.
      </Callout>

      <Stack gap={8}>
        <H2>Systems details that affect the result</H2>
        <Text>
          Add a KV cache. The current sampler recomputes the entire prefix every step, so reducing
          token count alone does not fix backbone cost. Keep the whole training path vectorized over
          `B·L·N`; never loop over examples or positions. Use EMA weights for generation, and report
          wall-clock sampling latency in addition to FID.
        </Text>
      </Stack>
    </Stack>
  );
}

function Experiments() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Decision-oriented experiment order</H2>
        <Table
          headers={["Order", "Experiment", "Comparison / metric", "Decision"]}
          rows={experimentRows}
          rowTone={["danger", "warning", "success", "info", "info", "neutral", "neutral", "warning"]}
          striped
        />
      </Stack>

      <Grid columns="1fr 1fr" gap={16}>
        <Stack gap={7}>
          <H3>Generation quality</H3>
          <Text>CIFAR-10 FID with a fixed sample count and evaluator</Text>
          <Text>Precision/recall or density/coverage, not FID alone</Text>
          <Text>Radial power-spectrum error by channel</Text>
          <Text>Image mean/contrast distributions</Text>
        </Stack>
        <Stack gap={7}>
          <H3>Mechanism diagnostics</H3>
          <Text>Loss by radius and diffusion timestep</Text>
          <Text>Late-prefix degradation versus teacher forcing</Text>
          <Text>Hermitian violation and imaginary reconstruction energy</Text>
          <Text>Latency split: Transformer versus denoising head</Text>
        </Stack>
      </Grid>

      <Callout tone="info" title="A useful ablation that preserves the number line">
        Before diffusion, try direct regression with a heteroscedastic full-covariance Gaussian.
        It already knows that nearby real values are nearby, models RGB/real-imag correlation, and
        gives a calibrated sampler. If it matches DiffLoss, the simpler head is the better model; if
        DiffLoss wins, the gain is evidence for conditional multimodality rather than merely removing
        cross-entropy buckets.
      </Callout>

      <Stack gap={8}>
        <H2>Open questions worth answering with data</H2>
        <Text>How multimodal is `p(coefficient | lower-frequency prefix)` at each radius?</Text>
        <Text>Does one coefficient per token destroy useful same-ring dependence or orientation symmetry?</Text>
        <Text>Should radial standardization be paired with explicit frequency loss weights?</Text>
        <Text>How many DDIM steps are needed in six dimensions, and does that vary by radius?</Text>
        <Text>Does generated-history error remain isotropic after normalization?</Text>
        <Text>Would 8–16 coefficient chunks dominate strict single-coefficient autoregression?</Text>
      </Stack>
    </Stack>
  );
}

export default function AFIGContinuousDiffusionReview() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState<Tab>("afig-continuous-review-tab", "design");

  return (
    <Stack gap={20} style={{ padding: 24, background: theme.bg.editor }}>
      <Stack gap={8}>
        <Row gap={10} align="center" wrap>
          <H1>AFIG continuous-token redesign</H1>
          <Pill active>Architecture review</Pill>
        </Row>
        <Text tone="secondary">
          Fourier coefficient autoregression with a conditional diffusion loss
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="3,264 → 544" label="AR steps with 6D tokens" tone="success" />
        <Stat value="6" label="Cartesian token dimensions" />
        <Stat value="4×" label="Paper's diffusion batch multiplier" tone="info" />
        <Stat value="10–25" label="Initial DDIM step range" tone="warning" />
      </Grid>

      <Row gap={8} wrap>
        <Pill active={tab === "design"} onClick={() => setTab("design")}>Design</Pill>
        <Pill active={tab === "representation"} onClick={() => setTab("representation")}>Representation</Pill>
        <Pill active={tab === "training"} onClick={() => setTab("training")}>Training</Pill>
        <Pill active={tab === "experiments"} onClick={() => setTab("experiments")}>Experiments</Pill>
      </Row>

      {tab === "design" ? <Design /> : null}
      {tab === "representation" ? <Representation /> : null}
      {tab === "training" ? <Training /> : null}
      {tab === "experiments" ? <Experiments /> : null}

      <Divider />
      <Text size="small" tone="tertiary">
        Sources: local AFIG code; Li et al.,{" "}
        <Link href="https://arxiv.org/abs/2406.11838">
          Autoregressive Image Generation without Vector Quantization
        </Link>
        ; official MAR implementation; Diffusers DDIM documentation; and 2025 FAR/NFIG follow-up
        work. No AFIG source files were modified.
      </Text>
    </Stack>
  );
}
