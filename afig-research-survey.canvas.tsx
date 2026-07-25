import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type View = "state" | "objective" | "shell" | "roadmap";

const runRows = [
  [
    "Additive position + FiLM + direct target",
    "36,162",
    "8.15",
    "Interrupted externally",
    "Broad structure; still blurry",
  ],
  [
    "Clean input + FiLM + direct target",
    "30,000",
    "8.19",
    "Completed",
    "Visually similar to additive",
  ],
  [
    "Clean input + direct target only",
    "30,000",
    "8.77",
    "Completed",
    "Visually similar; ~7% faster",
  ],
];

const currentConfigRows = [
  ["Data representation", "514 conjugacy-orbit representatives; 6D RGB real/imag"],
  ["Ordering", "Exact Euclidean radius, then angle"],
  ["Normalization", "Per-integer-radius Cholesky whitening"],
  ["Backbone", "10 layers × 768 width, 12 heads, causal, KV cached"],
  ["History features", "Physical polar log-amplitude + gated phase"],
  ["Diffusion target", "v-prediction, cosine DDPM schedule, 1,000 timesteps"],
  ["Timestep sampling", "Uniform discrete timesteps"],
  ["Diffusion loss", "Min-SNR γ=5 × radial amplitude weight α=0.5"],
  ["Denoiser", "3-block, width-768 AdaLN MLP"],
  ["Sampling", "20-step DDIM, η=0"],
  ["Conditioning", "Unconditional images; target frequency supplied directly"],
];

function SectionTabs({
  view,
  setView,
}: {
  view: View;
  setView: (view: View) => void;
}) {
  return (
    <Row gap={8} wrap>
      <Pill active={view === "state"} onClick={() => setView("state")}>
        Where we are
      </Pill>
      <Pill active={view === "objective"} onClick={() => setView("objective")}>
        Loss and timestep knobs
      </Pill>
      <Pill active={view === "shell"} onClick={() => setView("shell")}>
        Radial-shell Transformer
      </Pill>
      <Pill active={view === "roadmap"} onClick={() => setView("roadmap")}>
        Recommended roadmap
      </Pill>
    </Row>
  );
}

function StateView() {
  return (
    <Stack gap={18}>
      <Grid columns={4} gap={12}>
        <Stat value="514" label="Current AR coefficient steps" />
        <Stat value="135" label="Exact-radius shells" tone="info" />
        <Stat value="3.81" label="Mean coefficients per shell" />
        <Stat value="8" label="Maximum shell length" />
      </Grid>

      <Callout title="Current diagnosis" tone="warning">
        Position injection is not the main bottleneck. Removing additive position
        caused no obvious visual regression, and removing Transformer FiLM produced
        nearly the same samples while improving throughput. The remaining blur points
        more strongly toward the objective, sampler, and one-coefficient factorization.
      </Callout>

      <H2>Current system</H2>
      <Table
        headers={["Subsystem", "Active configuration"]}
        rows={currentConfigRows}
        striped
      />

      <H2>Matched positional ablations</H2>
      <Table
        headers={[
          "Variant",
          "Steps",
          "Throughput (steps/s)",
          "Status",
          "Visual read",
        ]}
        rows={runRows}
        columnAlign={["left", "right", "right", "left", "left"]}
        rowTone={["warning", "success", "success"]}
        striped
      />
      <Card>
        <CardHeader trailing={<Pill size="sm">Higher is faster</Pill>}>
          Training throughput by positional variant
        </CardHeader>
        <CardBody>
          <BarChart
            categories={["All-on", "Clean + FiLM", "Direct only"]}
            series={[{ name: "Training throughput", data: [8.15, 8.19, 8.77] }]}
            yMin={7.8}
            yMax={9.0}
            beginAtZero={false}
            valueSuffix=" steps/s"
            height={220}
            showValues
          />
          <Text size="small" tone="tertiary">
            x-axis: positional variant · y-axis: training throughput (steps/s).
            Source: local AFIG run logs, July 23–24, 2026. Rates are terminal or
            late-run progress rates.
          </Text>
        </CardBody>
      </Card>

      <Text tone="secondary">
        The terminal loss values are intentionally not ranked here: each is a noisy
        single diffusion batch, and the interrupted all-on run used a different
        cosine-schedule horizon.
      </Text>
    </Stack>
  );
}

function ObjectiveView() {
  return (
    <Stack gap={18}>
      <Callout title="Min-SNR and logit-normal are different knobs" tone="info">
        <Text>
          Current Min-SNR changes the <Text weight="semibold">loss weight</Text> after
          sampling a uniform timestep. Logit-normal usually changes the{" "}
          <Text weight="semibold">timestep sampling distribution</Text>. If samples
          are drawn from a non-uniform distribution without importance correction,
          the training objective itself changes.
        </Text>
      </Callout>

      <H2>Active weighting stack</H2>
      <Table
        headers={["Stage", "Current choice", "Effect", "Concern"]}
        rows={[
          [
            "Timestep draw",
            "Uniform t ∈ {0,…,999}",
            "Equal count per discrete diffusion time",
            "Not equal in log-SNR space",
          ],
          [
            "Diffusion weighting",
            "v-pred Min-SNR, γ=5",
            "Suppresses extreme low- and high-SNR examples",
            "May already concentrate strongly on middle SNR",
          ],
          [
            "Frequency weighting",
            "Radial power exponent α=0.5",
            "Expected-amplitude emphasis after whitening",
            "Maximum orbit weight is 50.5×; likely favors low frequency",
          ],
          [
            "Reduction",
            "Mean over active Cartesian components",
            "Handles self-conjugate masks",
            "Shell grouping must preserve per-coefficient normalization",
          ],
        ]}
        rowTone={["neutral", "info", "warning", "neutral"]}
        striped
      />

      <Grid columns={3} gap={12}>
        <Card>
          <CardHeader trailing={<Pill size="sm" active>Highest priority</Pill>}>
            Radial weighting
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Compare <Code>α=0</Code>, <Code>0.25</Code>, and current{" "}
                <Code>0.5</Code>.
              </Text>
              <Text size="small" tone="secondary">
                The current 50.5× maximum weight is the clearest objective-level
                explanation for good broad structure but weak high frequencies.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Timestep objective</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Compare uniform/no weighting, uniform/Min-SNR γ=5, and
                logit-normal sampling without Min-SNR.
              </Text>
              <Text size="small" tone="secondary">
                Do not stack both on the first pass; that obscures which mechanism
                helps.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Sampler ceiling</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Compare 20 versus 50 DDIM steps from the same trained model.
              </Text>
              <Text size="small" tone="secondary">
                This separates denoiser/discretization blur from representation or
                training blur.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Minimal objective survey</H2>
      <Table
        headers={["Run", "Timestep sampler", "Diffusion weight", "Radial α", "Question"]}
        rows={[
          ["A", "Uniform", "None", "0", "Unweighted reference"],
          ["B", "Uniform", "Min-SNR γ=5", "0", "Is Min-SNR helping without radial bias?"],
          ["C", "Logit-normal", "None", "0", "Does middle-time sampling help?"],
          ["D", "Uniform", "Min-SNR γ=5", "0.25", "Does mild amplitude emphasis help?"],
          ["E (current)", "Uniform", "Min-SNR γ=5", "0.5", "Existing low-frequency-heavy reference"],
        ]}
        rowTone={["neutral", "info", "info", "info", "warning"]}
        striped
      />

      <Divider />
      <H3>Other available or near-term knobs</H3>
      <Table
        headers={["Knob", "Current", "Useful alternatives", "Priority"]}
        rows={[
          ["Prediction target", "v", "ε; later x₀/EDM preconditioning", "Medium"],
          ["Noise schedule", "Cosine DDPM", "Zero-terminal-SNR; log-SNR-designed schedule", "Medium"],
          ["Denoiser depth", "3 MLP blocks", "5–8 blocks or shell Transformer", "Medium / architectural"],
          ["Diffusion draws", "1 per token", "2–4 independent (t, ε) draws", "Low after batch scaling"],
          ["History corruption", "None", "Small normalized Gaussian noise", "Later, for exposure bias"],
          ["Value transform", "Identity", "asinh", "Low until weighting is settled"],
          ["Flow matching", "Stub", "Rectified flow + logit-normal t", "Promising, separate phase"],
        ]}
        striped
      />
    </Stack>
  );
}

function ShellView() {
  const theme = useHostTheme();
  const stageStyle = {
    background: theme.fill.tertiary,
    border: `1px solid ${theme.stroke.tertiary}`,
    borderRadius: 6,
    padding: 12,
    minWidth: 150,
    flex: 1,
  };

  return (
    <Stack gap={18}>
      <Callout title="Recommended architectural experiment" tone="success">
        Keep the outer causal backbone initially, but replace the per-coefficient
        AdaLN MLP with a small bidirectional Transformer that jointly denoises every
        coefficient on one exact Euclidean shell.
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat value="135" label="Diffusion calls per image at 1 call/shell" tone="success" />
        <Stat value="3.81×" label="Fewer diffusion calls than 514 coefficients" tone="success" />
        <Stat value="1–8" label="Target sequence length per shell" />
        <Stat value="≤8" label="Cheap inner self-attention length" />
      </Grid>

      <Card>
        <CardHeader>Exact-radius shell-size distribution</CardHeader>
        <CardBody>
          <BarChart
            categories={["1 coefficient", "2 coefficients", "3–4 coefficients", "5–8 coefficients"]}
            series={[{ name: "Number of exact-radius shells", data: [2, 38, 77, 18] }]}
            valueSuffix=" shells"
            height={230}
            showValues
          />
          <Text size="small" tone="tertiary">
            x-axis: number of conjugacy-orbit representatives in one exact-r²
            shell · y-axis: shell count. Source: the current 32×32 radial codec;
            514 representatives grouped into 135 exact values of kx²+ky².
          </Text>
        </CardBody>
      </Card>

      <H2>Minimal shell-DiT dataflow</H2>
      <Row gap={10} align="stretch" wrap>
        <div style={stageStyle}>
          <Text weight="semibold">1. History context</Text>
          <Text size="small" tone="secondary">
            Run the existing causal backbone once. Gather the hidden state at each
            shell’s first coefficient, which sees only lower-radius shells.
          </Text>
        </div>
        <Text as="span" tone="tertiary" style={{ alignSelf: "center" }}>→</Text>
        <div style={stageStyle}>
          <Text weight="semibold">2. Noisy shell tokens</Text>
          <Text size="small" tone="secondary">
            Pad each shell to length 8. Input noisy 6D coefficients, target
            frequency features, one shared diffusion time, and a validity mask.
          </Text>
        </div>
        <Text as="span" tone="tertiary" style={{ alignSelf: "center" }}>→</Text>
        <div style={stageStyle}>
          <Text weight="semibold">3. Bidirectional denoiser</Text>
          <Text size="small" tone="secondary">
            Apply 2–4 small Transformer blocks with no causal mask inside the
            shell. Condition through AdaLN or cross-attention from history.
          </Text>
        </div>
        <Text as="span" tone="tertiary" style={{ alignSelf: "center" }}>→</Text>
        <div style={stageStyle}>
          <Text weight="semibold">4. Joint shell output</Text>
          <Text size="small" tone="secondary">
            Predict v or ε for all shell coefficients, enforce component masks,
            then append the completed shell to the outer KV cache in angle order.
          </Text>
        </div>
      </Row>

      <H2>Concrete first implementation</H2>
      <Table
        headers={["Element", "First version", "Reason"]}
        rows={[
          ["Grouping key", "Exact integer r² = kx² + ky²", "No floating-point grouping ambiguity"],
          ["Outer backbone", "Keep coefficient-level 10×768 Transformer", "Avoid changing two major systems at once"],
          ["Shell context", "One z from the shell-start hidden state", "Strictly lower-radius causal context"],
          ["Inner width/depth", "256–384 width, 2–4 blocks, 4–6 heads", "Sequences are at most length 8; width 768 is unnecessary"],
          ["Inner attention", "Bidirectional within shell", "All same-radius targets are generated jointly"],
          ["Diffusion time", "One t per shell; independent Gaussian noise per value", "Defines a coherent joint shell denoising problem"],
          ["Target identity", "Functional 2D frequency features per shell token", "Preserves exact angle and exceptional-mode identity"],
          ["Loss reduction", "Sum per coefficient, normalize across all active coefficients", "Avoid overweighting short shells"],
          ["Generation", "20 denoising steps × 135 shells", "2,700 head evaluations versus 10,280 currently"],
        ]}
        striped
      />

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader trailing={<Pill size="sm" active>Upside</Pill>}>
            Why this may improve quality
          </CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>Same-radius angular modes can coordinate jointly.</Text>
              <Text>RGB and phase relationships extend across coefficients, not only channels.</Text>
              <Text>The denoiser no longer assumes conditional independence within a ring.</Text>
              <Text>Generation requires substantially fewer expensive diffusion loops.</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Watch carefully</Pill>}>
            Main risks
          </CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>Equal-shell averaging would badly overweight DC and short shells.</Text>
              <Text>A single shell context may bottleneck angular history information.</Text>
              <Text>Sequential cache insertion creates an arbitrary within-shell order.</Text>
              <Text>Shell-level backbone compression should be a later, separate ablation.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function RoadmapView() {
  return (
    <Stack gap={18}>
      <Callout title="Recommended order" tone="info">
        Stay unconditional until the model produces recognizably coherent,
        reasonably sharp CIFAR-10 samples. Class conditioning and CFG can improve
        scores while masking whether the unconditional density model is actually
        healthy.
      </Callout>

      <Table
        headers={["Phase", "Work", "Decision criterion"]}
        rows={[
          [
            "1. Objective sanity",
            "Radial α ∈ {0, 0.25, 0.5}; Min-SNR versus none; 20 versus 50 DDIM steps",
            "Identify whether blur is loss weighting or sampler limited",
          ],
          [
            "2. Shell-DiT prototype",
            "135 exact-radius groups; small bidirectional diffusion Transformer",
            "Sharper structure or better likelihood proxy at matched compute",
          ],
          [
            "3. Timestep distribution",
            "Uniform versus logit-normal/log-SNR sampling; do not initially stack with Min-SNR",
            "Better convergence and high-frequency reconstruction",
          ],
          [
            "4. Robust autoregression",
            "Small history corruption or short sampled-prefix training",
            "Reduce teacher-forcing versus generation gap",
          ],
          [
            "5. Larger generative reformulation",
            "Rectified flow / EDM-style preconditioning; optional shell-level backbone",
            "Only if DDPM objective remains limiting",
          ],
          [
            "6. Class conditioning",
            "Class embeddings, ~10% condition dropout, CFG scale sweep",
            "After unconditional quality is credible",
          ],
        ]}
        rowTone={["warning", "success", "info", "neutral", "neutral", "neutral"]}
        striped
      />

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Immediate experimental matrix</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Cheap current-head survey</Text>
              <Text>Short matched runs for radial α=0 and α=0.25.</Text>
              <Text>For α=0, compare Min-SNR γ=5 against no SNR weighting.</Text>
              <Text>Render both 20- and 50-step samples at validation.</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Parallel engineering track</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">Implement behind a grouping flag</Text>
              <Text><Code>grouping=coefficient</Code> remains the reference path.</Text>
              <Text><Code>grouping=exact_radius_shell</Code> activates padded shell targets.</Text>
              <Text>Do not add class conditioning or 2D RoPE in the same change.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Divider />
      <Text tone="secondary">
        Bottom line: the strongest next architectural bet is the exact-radius
        shell Transformer head. The strongest near-term objective check is to
        reduce or remove radial amplitude weighting before attributing blur to
        model capacity.
      </Text>
    </Stack>
  );
}

export default function AFIGResearchSurvey() {
  const theme = useHostTheme();
  const [view, setView] = useCanvasState<View>("afig-survey-view", "state");

  return (
    <Stack
      gap={18}
      style={{
        maxWidth: 1180,
        margin: "0 auto",
        padding: 24,
        color: theme.text.primary,
      }}
    >
      <Stack gap={6}>
        <H1>AFIG research survey</H1>
        <Text tone="secondary">
          Current evidence, loss-weighting options, and a concrete exact-radius
          shell Transformer proposal for unconditional CIFAR-10 generation.
        </Text>
      </Stack>
      <SectionTabs view={view} setView={setView} />
      <Divider />
      {view === "state" && <StateView />}
      {view === "objective" && <ObjectiveView />}
      {view === "shell" && <ShellView />}
      {view === "roadmap" && <RoadmapView />}
    </Stack>
  );
}
