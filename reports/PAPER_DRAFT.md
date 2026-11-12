# Composition, not condition: what 6.8 million deliveries can and cannot say about pitch wear

**Draft for conference submission — not yet formatted for a venue.**

Aravind R, Rishab H R, Ujwal S Nair
Department of Electronics and Communication Engineering, RVCE, Bengaluru

---

## Abstract

Pitch wear is a first-order tactical variable in cricket and there is no public
measurement of it. The natural proxy is the within-match change in the relative
effectiveness of spin and pace, estimated from ball-by-ball data with match fixed
effects. Using 6.8 million deliveries from 23,241 matches, we show that this
proxy is precise, stable and not identified as wear. Four validity checks fail.
Within-innings permutation nulls are positively biased by about a fifth, so the
usual randomisation test overstates significance. The estimate reverses sign
between phases, which no uniform surface process predicts. The runs channel
disappears when innings boundaries are trimmed, and the per-match estimate has
zero cross-innings persistence and negligible predictive value. We then test the
most natural identification strategy: the second new ball at over 80 in Test
cricket. Across all deliveries the pace dot-ball share rises by 1.5 percentage
points at the boundary. Within the same bowler and innings it falls by 2.1
points, because the new ball also brings the frontline seamers. The apparent ball
effect is selection. A second natural experiment, the October 2011 two-new-ball
rule in One Day Internationals, leaves the age gradient unchanged. We conclude
that within-innings gradients in open ball-by-ball data are dominated by
composition rather than condition, and we give a short audit that any pitch-wear
claim should pass.

---

## 1. Introduction

Cricket captains read the pitch before they read the team sheet. A surface that
starts damp and quick can become slow and low by the third afternoon; a dry
surface can take turn from the first hour and grip harder later. The state of the
pitch changes who should bowl, what a par score is, and whether winning the toss
matters. Yet no public data set records the pitch. Broadcasters measure soil
moisture, hardness and grass cover with instruments that are not shared, and the
governing body does not publish surface readings. Any quantitative claim about
pitch behaviour therefore has to be inferred from play.

The most attractive inference is also the most dangerous. If a pitch wears, spin
should become relatively more effective than pace as the match proceeds. That
suggests a simple estimator: regress an outcome such as runs per ball or dot-ball
share on ball age, a spin indicator, and their interaction, with match fixed
effects to absorb venue, weather, teams and date. A positive age-by-spin
coefficient on dot-ball share then reads as wear. Variants of this design appear
whenever analysts discuss "the pitch getting better for spin", and the same
logic underlies proposed pitch-wear indices.

This paper asks whether that estimator measures what it claims. The answer is no,
and the reasons are instructive. The problem is not statistical power: with 6.8
million deliveries the coefficient is estimated to two significant figures and is
stable across rolling windows. The problem is that the coefficient is a composite
of at least four processes that move together with ball age: the surface, the
ball, the phase of the innings, and which bowler is bowling. Open ball-by-ball
data contain no variation that separates them cleanly, and the one boundary that
looks like a natural experiment — the second new ball — turns out to be a
selection trap.

Our contributions are:

1. **A large-scale audit of the standard estimator.** We estimate the age-by-spin
   gradient on 5.1 million legal spin and pace deliveries aggregated to 855,659
   over-by-style cells, with match, innings and bowler fixed effects and
   match-clustered standard errors. The pooled coefficient replicates and is
   stable, so the problem is not the sample.

2. **A diagnostic for permutation inference.** Randomising bowling style within
   innings produces a null distribution that is not centred on zero. About a
   fifth of the observed dot-ball gradient is reproduced by random allocation,
   because the phase structure of an innings is itself informative. A permutation
   test that assumes a zero-centred null is anti-conservative.

3. **A phase decomposition that reverses.** The pooled gradient is an average of
   offsetting phase effects: the powerplay contributes the largest positive term
   and the T20 death overs the largest negative one. A physical wear process
   cannot change sign at the same ball age, so the pooled coefficient is a
   weighted average of different regimes rather than a rate.

4. **A failed natural experiment, reported as a result.** The second new ball at
   over 80 appears to raise the pace dot-ball share by 1.5 percentage points.
   Holding the bowler and innings fixed, the same boundary lowers it by 2.1
   points. The naive estimate is the arrival of the frontline seamers, not the
   arrival of a new ball. This is the composition problem in its most visible
   form, and it is the paper's central caution.

5. **A ball-age regime test.** The October 2011 two-new-ball rule halves the rate
   at which an ODI ball ages per over of play. If the smooth gradient were ball
   ageing, it should halve. It does not; the ODI gradient is essentially
   unchanged, and a T20 control shows no common shock of the required size.

6. **What survives.** The per-match index has zero cross-innings persistence and
   negligible predictive value. The one surface-scale signal that survives is the
   across-innings contrast in Tests, where spin's dot advantage is 1.1 percentage
   points higher in later innings of the same match.

The practical message is a checklist rather than a metric. Before a within-innings
gradient is called wear, it should survive a within-innings permutation null, a
phase decomposition, an innings-boundary trim, a bowler-fixed comparison and a
persistence test. In this data set the dot-ball gradient passes the boundary trim
and survives the corrected permutation null in T20 and ODI, but it fails the phase
check, cannot be given a bowler-fixed interpretation, and fails the persistence
test.

## 2. Related work

**Pitch measurement.** The direct literature is small and instrument-based. Adams
et al. [1] measured bounce, hardness and friction on 92 County Championship
pitches and compared the readings with umpires' ratings; James et al. [2]
characterised the playing performance of county surfaces. Both establish that
surface properties are measurable, and both require on-site equipment that no
public archive provides. Our work is the complement: what can be inferred
without instruments, and what cannot.

**Prediction from open data.** A larger literature predicts results rather than
surfaces. Morley and Thomas [3] model home advantage in English one-day cricket,
Sood and Willis [4] estimate the toss effect across more than 41,000 matches,
Chakraborty et al. [5] and Shah et al. [6] compare classifiers for T20 outcomes,
and Bhatnagar and Bhatnagar [7] combine automated machine learning with pitch
reports to classify first-innings scores. Venue behaviour profiles — grounds
ranked by average score, wicket rate or spin share — are a common construct in
this literature and in practice. They are descriptive and, because no surface
measurements are public, unvalidated. Our paper asks what such a profile can be
built from, and shows that the within-match version is not identified as surface
change.

**Composition and inference.** Three methodological ideas carry the argument.
First, fixed effects remove level differences but not composition: when units are
heterogeneous, a within-group coefficient is a weighted average of unit-specific
effects, which is the critique of two-way fixed-effects estimators under
heterogeneous effects [8]. Second, randomisation inference is valid only when
treatment assignment is exchangeable within the randomisation unit [9], [10],
and the modern formulation makes the assignment mechanism explicit [11]; when the
unit is an innings and the outcome depends on over number, the null is not
centred on zero. Third, sport provides bundled natural experiments. A boundary
design identifies a discontinuity in treatment, not the component of the bundle
that causes it [12], so a new-ball discontinuity is evidence about the new ball
*and* the attack and the field. Cluster-robust inference at match level follows
[13]; the difference-in-differences design and its serial-correlation caveat
follow [14].

## 3. Data

**Ball-by-ball archive.** We use the Cricsheet archive of ball-by-ball scorecards
[15], which covers international and major domestic competitions. The extract used here
contains **6,776,668 deliveries from 23,241 matches** played between December
2001 and November 2026, across T20, ODI and Test cricket and both genders.

**Bowling style.** Cricsheet records the bowler's name but not the bowling style.
We attach a style with a conservative, evidence-based protocol: a player profile
is accepted only when the name match is unambiguous, the profile's national team
agrees with the team the player represents in Cricsheet, and the profile's
bowling style agrees with the style class implied by the player's career record.
The protocol resolves 77.4% of deliveries. The remaining **22.6% have an unknown
style and are excluded**, which is a real limitation rather than a rounding
error; it is reported in every table.

**Analysis sample.** Restricting to legal deliveries with a known style leaves
**5,116,165 deliveries**. These aggregate to **855,659 over-by-style cells**,
where a cell is a (match, innings, over, style) combination with at least three
legal balls. Cells are weighted by their ball count in all regressions.

**Outcomes.** Four outcomes are used because none is a clean pitch measure:
runs per ball, wicket rate, dot-ball share, and boundary share. If wear is real,
spin's relative performance should improve with ball age on all four: fewer runs,
more wickets, more dots, fewer boundaries.

**Rule dates.** The second new ball in Tests is available after 80 overs
throughout the sample. In ODIs, a mandatory ball change at the start of the 35th
over applied before 1 October 2011 [17], when it was replaced by two new balls
per innings, one from each end, used in alternate overs [16]. We use those dates
as given and verify the mechanics of both regimes: the pace share of deliveries
jumps from 40% to 74% at over 80 in Tests, and each post-2011 ODI ball is in play
for at most about 25 overs.

## 4. Estimator

For a cell $c$ in match $m$, innings $i$, over $t$ and bowling style $s$, we
estimate

$$
y_{c} = \alpha_{m} + \beta\,a_{c} + \gamma\,s_{c} + \delta\,(a_{c}\times s_{c})
+ \mathbf{x}_{c}'\theta + \varepsilon_{c},
$$

where $a_{c}=t+0.5$ is ball age in overs, $s_{c}=1$ for spin, $\alpha_{m}$ is a
match (or match-by-innings, or match-by-innings-by-bowler) fixed effect, and
$\mathbf{x}_{c}$ contains wickets lost, the run rate before the over and the mean
balls faced by the batter. Cells are weighted by legal balls and standard errors
are one-way cluster-robust at match level [13]. The coefficient of interest is
$\delta$, the age-by-spin interaction.

The interpretation of $\delta$ deserves care. A negative $\delta$ on runs per ball
means spin concedes relatively fewer runs as the ball ages. That is consistent
with wear, but also with a change in which overs spinners bowl, a softening ball,
a change in field settings, or batters adapting. The rest of the paper is a
sequence of tests designed to remove those alternatives.

## 5. The gradient is precise and stable

Table 1 reports $\delta$ under the match fixed-effects specification. The
dot-ball coefficient is positive in all three formats and significant at
$p<0.001$; the runs coefficient is negative and significant in T20 and ODI and
marginal in Test; the wicket-rate coefficient is null in T20 and Test and has the
wrong sign in ODI; the boundary coefficient is null except for a weak negative in
Test.

**Table 1: Age-by-spin coefficient by format and outcome (match fixed effects)**

| Format | Runs per ball | Dot-ball share | Wicket rate | Boundary share |
| --- | --- | --- | --- | --- |
| T20 | −0.00198 (p<0.001) | +0.00151 (p<0.001) | +0.00009 (p=0.187) | −0.00012 (p=0.285) |
| ODI | −0.00131 (p<0.001) | +0.00146 (p<0.001) | −0.00018 (p<0.001) | +0.00003 (p=0.511) |
| Test | −0.00012 (p=0.051) | +0.00013 (p<0.001) | −0.00000 (p=0.421) | −0.00003 (p=0.035) |

The dot-ball channel is the only one that is positive in every format and never
changes sign. It is also stable over time: refitting on four expanding windows
gives train coefficients between +0.00150 and +0.00155 in T20 and between +0.00144
and +0.00151 in ODI, with held-out coefficients of the same order. A reader who
stopped here would be entitled to think the estimator works. The remaining
sections explain why that reader would be wrong.

## 6. Four validity checks

### 6.1 The permutation null is biased

The standard placebo reallocates spin and pace labels at random within each
innings and refits the same specification [9], [10]. If the coefficient were
driven by style, the observed value should lie far in the tail of the null. It
does, but the null is not where it should be. Table 2 reports 100 draws per cell.

**Table 2: Within-innings permutation placebo (100 draws)**

| Format | Outcome | Observed | Null mean | Null sd | p |
| --- | --- | --- | --- | --- | --- |
| T20 | Dot-ball share | +0.00138 | +0.00027 | 0.00012 | <0.01 |
| T20 | Runs per ball | −0.00182 | −0.00182 | 0.00041 | 0.52 |
| ODI | Dot-ball share | +0.00140 | +0.00030 | 0.00006 | <0.01 |
| ODI | Runs per ball | −0.00108 | −0.00050 | 0.00014 | <0.01 |
| Test | Dot-ball share | +0.00012 | +0.00009 | 0.00002 | 0.06 |
| Test | Runs per ball | −0.00009 | −0.00018 | 0.00005 | 0.97 |

Two facts stand out. First, the null mean is positive for dot-ball share in every
format: randomising styles inside an innings still produces about a fifth of the
observed gradient, because spinners and seamers are not interchangeable across
overs even after the labels are shuffled. A test that assumes a zero-centred null
will overstate significance. Second, once the correct null is used, the T20 runs
effect is indistinguishable from noise ($p=0.52$). The channel that most papers
would report is the one that fails first.

### 6.2 The coefficient reverses by phase

Table 3 re-estimates the specification inside each phase. The heterogeneity is
not a refinement; it changes the interpretation.

**Table 3: Age-by-spin coefficient inside each phase**

| Format | Phase | Dot-ball share | Runs per ball |
| --- | --- | --- | --- |
| T20 | Powerplay | +0.00752 (p<0.001) | −0.02589 (p<0.001) |
| T20 | Middle | +0.00114 (p=0.010) | +0.00408 (p=0.002) |
| T20 | Death | −0.00623 (p<0.001) | +0.01396 (p=0.010) |
| ODI | Powerplay | +0.00240 (p=0.024) | −0.01144 (p<0.001) |
| ODI | Middle | +0.00109 (p<0.001) | +0.00064 (p=0.056) |
| ODI | Death | +0.00112 (p=0.294) | −0.00410 (p=0.195) |
| Test | New ball | −0.00101 (p<0.001) | +0.00112 (p=0.011) |
| Test | Middle | +0.00027 (p=0.004) | −0.00028 (p=0.173) |
| Test | Old ball | −0.00003 (p=0.761) | +0.00041 (p=0.085) |

The T20 powerplay alone carries a dot-ball coefficient five times the pooled
value, and the T20 death overs carry the opposite sign. A surface cannot wear in
one direction for six overs and the other direction for the next five. The pooled
number is a weighted average of tactical regimes, and the weights are set by how
often each phase occurs, not by anything physical.

The middle overs are the most surface-like regime: dot share rises for spin in all
three formats, but so does the boundary share in T20 and ODI. That is an attacking
signature — beat the bat or disappear — and it is equally consistent with a
field-setting change. The data cannot separate the two.

### 6.3 The runs effect is an innings-boundary effect

Trimming the first and last over of every innings removes the T20 runs
coefficient entirely: −0.00198 becomes −0.00010 ($p=0.87$). The dot-ball
coefficient survives the same trim (+0.00142), as it does the established-bowler
restriction (+0.00162) and the middle-overs-only sample (+0.00114). The runs
channel was carried by the overs in which pace bowls and spin rarely does, which
is the composition problem again.

### 6.4 The per-match index does not persist

If the coefficient measured a surface, the two innings of the same match should
produce similar estimates. They do not. Correlating the innings-level coefficient
across the two innings gives Pearson $r=-0.003$ in T20, $-0.047$ in ODI and
$+0.030$ in Test (all $n\ge763$). The one nominally significant value is negative
and tiny.

A wear term should also help predict held-out play. Adding the age-by-spin term to
a ridge model of over-level outcomes changes weighted mean squared error by
between −0.007% and +0.153% depending on format and outcome. The largest gain, on
ODI dot-ball share, is statistically distinguishable from zero and economically
irrelevant, and three of the twelve outcome-format combinations get slightly
worse.

Classical measurement error explains part of this: a single innings supplies few
spin overs, so the innings-level slope is noisy and attenuates towards zero. But
attenuation is not a rescue. It means the estimator cannot measure an individual
pitch at the sample sizes cricket provides, and the population coefficient must be
interpreted as a league-wide average of a mixture, not as a physical rate.

## 7. The new ball is a selection trap

The tests above reject a uniform wear rate but do not say what the gradient is.
The leading alternative is ball condition. The ball softens and scuffs as an
innings proceeds, and that changes the same outcomes used to infer wear. The
obvious way to separate the two is the second new ball: at over 80 in a Test
innings the fielding side may take a new ball, and in practice does. The ball is
renewed while the surface keeps ageing. A jump in outcomes at that boundary,
within the same innings and controlling for a smooth trend in over number, would
look like a ball-condition effect.

We estimate an event study in two-over bins relative to overs 78-79, on 162,467
Test deliveries in overs 72-87. The naive specification holds the match fixed and
lets the bowler mix change. The within-bowler specification also holds the bowler
and innings fixed, and restricts the sample to bowler-innings units that bowl on
both sides of the boundary. Standard errors are clustered by match. The
comparison is a check on what the boundary identifies, not a regression-
discontinuity estimate of the new ball itself [12].

**Table 4: Dot-ball share around the second new ball, relative to overs 78-79**

| Bin | All deliveries (naive) | Same bowler, same innings |
| --- | --- | --- |
| Pace, 80-81 | +0.0148 (p=0.031) | −0.0210 (p=0.044) |
| Pace, 82-83 | +0.0035 (p=0.622) | −0.0293 (p=0.024) |
| Pace, 84-85 | +0.0029 (p=0.685) | −0.0305 (p=0.044) |
| Pace, 86-87 | −0.0045 (p=0.541) | −0.0449 (p=0.020) |
| Spin, 80-81 | −0.0100 (p=0.129) | −0.0344 (p<0.001) |
| Spin, 82-83 | −0.0022 (p=0.795) | −0.0379 (p<0.001) |
| Spin, 84-85 | +0.0021 (p=0.818) | −0.0364 (p=0.005) |
| Spin, 86-87 | −0.0018 (p=0.838) | −0.0521 (p<0.001) |

The sign reverses. Across all deliveries, pace's dot-ball share rises by 1.5
percentage points in the first two overs with the new ball. Within the same
bowler, it falls by 2.1 points and keeps falling. The naive jump is the arrival
of the frontline seamers: the pace share of deliveries rises from 40% before over
80 to 74% after it. The ball is new, but so is the attack.

Figure 1 shows the two paths. The grey line is the naive series, with a visible
bump at the boundary. The blue line is the within-bowler series, which declines
smoothly through the same window. Whatever the decline reflects — batters
settling, spell fatigue, or a harder ball that finds the edge rather than the
middle — it is not a new-ball bonus. The boundary cannot be used to identify ball
condition, and the same composition logic applies to the wear estimator.

**Figure 1.** Event study around the second new ball. Grey: all deliveries with
match fixed effects. Blue: within the same bowler and innings. Vertical line at
over 80. Bars are 95% confidence intervals from match-clustered standard errors.

A second test asks whether the smooth gradient is ball ageing. From 1 October
2011 each ODI innings has used two new balls, one from each end, in alternate
overs [16]. Each ball is in play for at most about 25 overs instead of 50,
so ball age accrues at half the rate per over of play, while the pitch wears at
the same rate. If the gradient were driven by ball ageing it should halve.
Table 5 gives the answer, with T20 as a control for common period effects. The
design is a difference-in-differences across format-era strata, estimated with
separate within-format fits [14].

**Table 5: Age-by-spin coefficient for dot-ball share before and after the
two-new-ball rule**

| Stratum | Coefficient (SE) |
| --- | --- |
| ODI before October 2011 | +0.00129 (0.00020) |
| ODI after October 2011 | +0.00138 (0.00010) |
| T20 before October 2011 | +0.00205 (0.00085) |
| T20 after October 2011 | +0.00146 (0.00016) |
| Difference-in-differences | +0.00069 (0.00089), p=0.44 |
| Predicted ODI change if ball ageing | −0.00064 |

The ODI gradient did not fall. It rose by 0.0001 against a prediction of
−0.0006, and the difference-in-differences is positive with an interval that spans
zero. The test is not decisive: the control format is imperfect, the rule came in
a package with other changes, and the interval is wide. It nevertheless gives no
support to the view that the smooth gradient is ball ageing.

Taken with the event study, the picture is that the ball is not the source of the
aggregate jump at renewal and not the driver of the smooth gradient. The
within-innings coefficient is neither a uniform wear rate nor a ball-age rate.

## 8. What survives

One result has the shape of a surface effect. Within the same match, spin's
dot-ball advantage is 1.07 percentage points higher in the second and later
innings than in the first (p<0.001), with a smaller positive contrast
for wicket rate (+0.00113, p=0.007) and boundary share (+0.00284, p=0.004). The
comparison is across innings of the same match, so venue, weather and teams are
held fixed, and the surface has had more time to wear than any within-innings
window allows.

It is not proof. Later innings differ in batting side, light, declaration
pressure and follow-on selection. But it is the only contrast in the data that is
both positive on multiple channels and consistent with a slow surface process,
and it points to the right scale: innings, not overs.

The contrast also explains why the within-innings estimator fails. Surface change
over 20 to 50 overs is small relative to the composition changes that occur over
the same window. The estimator is asked to detect a weak signal while four
stronger signals move with the same regressor, and it has no instrument that
separates them.

## 9. An audit for pitch-wear claims

We propose five checks, in increasing order of cost. Each is cheap on
ball-by-ball data and each corresponds to a failure mode documented above.

1. **Report the permutation null, not just the p-value.** Randomise style within
   innings, refit, and report the null mean. If the null mean is a non-trivial
   fraction of the observed coefficient, the claim is partly mechanical.
2. **Report the estimate by phase.** A wear rate should not change sign between
   phases at the same ball age. If it does, report the weighted average as a
   mixture, not as a rate.
3. **Trim innings boundaries.** The first and last overs are where style
   composition is most extreme, and they can carry an entire runs effect.
4. **Hold the bowler fixed.** For any discontinuity design, estimate the effect
   within bowler-by-innings. If the sign reverses, the design measures selection.
5. **Test persistence and prediction.** Correlate the innings-level estimate
   across innings of the same match, and measure the out-of-sample gain. A
   measurement that does not persist is not a measurement of the pitch.

A claim that passes all five is a genuine surface estimate. A claim that reports
only the pooled coefficient and a confidence interval is reporting a mixture.

## 10. Threats to validity

**No physical measurement.** The paper cannot validate against soil moisture,
hardness or grass cover, because those data are not public. Every result is a
statement about what can be inferred from play, not about the pitch itself. This
is the central limit and we do not claim otherwise.

**Style coverage.** 22.6% of deliveries have no resolved bowling style and are
excluded. The protocol is conservative by design, so the excluded set is not
random: it over-represents players with ambiguous names or incomplete profiles.
If style misclassification is correlated with over number, the estimated gradient
is biased. The established-bowler restriction, which keeps the 1,491 bowlers with
at least 300 legal balls, leaves the dot-ball result intact, which bounds the
problem but does not eliminate it.

**Bundled treatments.** The second new ball changes the ball, the bowler, the
field and the batter's intent. The within-bowler design removes the bowler but not
the tactics. We therefore interpret the within-bowler estimate as "not a
ball-condition bonus", not as a precise negative causal effect of a new ball.

**Rule changes are packages.** The 2011 ODI rule also changed powerplay timing
and field restrictions. The T20 control absorbs the common change — the T20
gradient itself fell by 0.0006 over the same period — but it cannot remove
ODI-specific changes.

**Functional form.** Event studies around a boundary are sensitive to how the
trend in over number is modelled. The design here uses bin indicators rather
than a fitted trend, so the contrast does not depend on a functional form for
age, and the within-bowler estimate is stable across specifications: −0.0192
without controls, −0.0210 with the wickets control used in the tables, and
−0.0211 with wickets and run rate.

**Multiple outcomes.** Four outcomes and three formats generate twelve headline
coefficients. We treat dot-ball share as the primary channel because it is the
only one with a consistent sign, and we report the others without correction,
which is a descriptive choice rather than a confirmatory one.

## 11. Conclusion

The appeal of ball-by-ball pitch wear is that it needs no instruments. The
problem is that the data contain no clean variation in the thing being measured.
What looks like wear is a mixture of phase, bowler selection, ball condition and
batter adaptation, and the mixture changes with the sample. In 6.8 million
deliveries the resulting coefficient is precise, stable and not identified.

The most useful result is the failed natural experiment. The second new ball
raises the aggregate pace dot rate by 1.5 percentage points and lowers the
same-bowler rate by 2.1 points. If a boundary this sharp can be pure selection,
then a smooth gradient across 50 overs can be too. Analysts who report
within-innings wear should report the composition audit alongside it, or state
plainly that they are describing a mixture.

The constructive path is to move the measurement to the scale where the signal
is larger than the confounds. Across innings, in the same match, spin's dot
advantage is higher by about one percentage point in Tests. That is small, but it
is the right order of magnitude for a surface process and it is the quantity a
pitch-wear index should try to predict.

---

## Reproducibility

All code and derived artifacts are in the project repository.

| Step | Command | Main output |
| --- | --- | --- |
| Build deliveries | `python src\build_ball_dataset.py` | `data/interim/balls.parquet` |
| Fit specifications | `python src\pitch_dynamics.py` | `wear_specifications.csv`, `wear_placebo.json` |
| Robustness | `python src\wear_robustness.py` | `wear_by_phase.csv`, `wear_robustness.csv` |
| Validation | `python src\validate_wear.py` | `wear_persistence.csv`, `wear_predictive_gain.csv`, `wear_rolling_coefficients.csv` |
| Identification | `python src\identify_wear.py all` | `wear_new_ball_event.csv`, `wear_two_new_ball.csv` |
| Figures | `python src\make_dynamics_figures.py` | `reports/figures/fig1`–`fig6` |

## References

[1] W. A. Adams, S. W. Baker, D. M. James, and R. J. Young, "Measuring and
modelling the bounce and pace of County Championship cricket pitches,"
*International Turfgrass Society Research Journal*, vol. 10, pp. 1021–1026, 2005.

[2] D. M. James, M. J. Carré, and S. J. Haake, "The playing performance of county
cricket pitches," *Sports Engineering*, vol. 7, no. 1, pp. 1–14, 2004.

[3] B. Morley and D. Thomas, "An investigation of home advantage and other
factors affecting outcomes in English one-day cricket matches," *Journal of
Sports Sciences*, vol. 23, no. 3, pp. 261–268, 2005.

[4] G. Sood and D. Willis, "Fairly random: The impact of winning the toss on the
probability of winning," arXiv:1605.08753, 2016.

[5] S. Chakraborty et al., "Cricket data analytics: Forecasting T20 match winners
through machine learning," *International Journal of Knowledge-Based and
Intelligent Engineering Systems*, vol. 28, no. 1, pp. 73–92, 2024,
doi: 10.3233/KES-230060.

[6] S. A. A. Shah, Q. Zaman, D. Wasim, J. Allohibi, A. A. Alharbi, and
M. Shabbir, "Optimal model for predicting highest runs chase outcomes in T-20
international cricket using modern classification algorithms," *Alexandria
Engineering Journal*, vol. 114, pp. 588–598, 2025.

[7] M. Bhatnagar and M. Bhatnagar, "Analyzing key factors influencing IPL cricket
scores using explainability and multimodal data," *Journal of Quantitative
Analysis in Sports*, vol. 21, no. 3, pp. 253–267, 2025,
doi: 10.1515/jqas-2025-0006.

[8] C. de Chaisemartin and X. D'Haultfœuille, "Two-way fixed effects estimators
with heterogeneous treatment effects," *American Economic Review*, vol. 110,
no. 9, pp. 2964–2996, 2020.

[9] R. A. Fisher, *The Design of Experiments*. Edinburgh, U.K.: Oliver & Boyd,
1935.

[10] P. R. Rosenbaum, *Observational Studies*, 2nd ed. New York, NY, USA:
Springer, 2002.

[11] S. Athey and G. W. Imbens, "The econometrics of randomized experiments," in
*Handbook of Econometrics*, vol. 1, Amsterdam, The Netherlands: Elsevier, 2017,
pp. 73–140.

[12] G. W. Imbens and T. Lemieux, "Regression discontinuity designs: A guide to
practice," *Journal of Econometrics*, vol. 142, no. 2, pp. 615–635, 2008.

[13] A. C. Cameron, J. B. Gelbach, and D. L. Miller, "Robust inference with
multiway clustering," *Journal of Business & Economic Statistics*, vol. 29,
no. 2, pp. 238–249, 2011.

[14] M. Bertrand, E. Duflo, and S. Mullainathan, "How much should we trust
differences-in-differences estimates?" *Quarterly Journal of Economics*,
vol. 119, no. 1, pp. 249–275, 2004.

[15] Cricsheet. Ball-by-ball cricket data archive. [Online]. Available:
https://cricsheet.org

[16] International Cricket Council. "David Richardson answers your questions."
4 October 2011. [Online]. Available:
https://www.icc-cricket.com/news/david-richardson-answers-your-questions

[17] International Cricket Council. "Amendments to playing conditions take
effect from 1 October." 6 April 2010. [Online]. Available:
https://www.icc-cricket.com/news/amendments-to-playing-conditions-take-effect-from-1-october

## Venue fit

The paper is a measurement-and-identification study with a large observational
sample and a negative central result. It fits workshops and tracks that value
rigour over a positive predictive headline:

- **MLSA (Machine Learning and Sports Analytics) workshop at ECML-PKDD** — short
  format, methods-and-application mix, strong fit for the audit framing.
- **IEEE BigData, Sports Analytics track** — longer format, accommodates the full
  event study and the rule-change test.
- **Journal of Sports Analytics** — rolling submission, appropriate if the
  related-work section is expanded.

The main revision risk is the absence of physical pitch data. The mitigation is
to frame the contribution as identification, not measurement, which is how the
draft is written.
