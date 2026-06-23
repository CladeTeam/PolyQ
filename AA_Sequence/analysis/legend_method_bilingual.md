# Supplementary Figure 2 — Legend & Methods

---

## Supplementary Figure 2

**ESM2-based species-adapted scoring identifies *Chlamydomonas*-compatible
exogenous polyQ scaffolds.**

Schematic of the species-adapted scoring strategy used to prioritise
polyQ-containing proteins for functional testing in *Chlamydomonas*.

**(a) Model construction.** The pretrained ESM2-650M protein language model
(base) is fine-tuned on the *C. reinhardtii* and *O. sativa* proteomes
separately via masked language modelling, producing Chlamydomonas-adapted and
rice-adapted model variants. Each model is trained for 3 epochs on its
respective proteome via continued masked language modelling.

**(b) Candidate collection and scoring.** PolyQ-containing proteins (≥10
consecutive glutamine residues) are collected from the proteomes of
*C. reinhardtii* and *O. sativa*. For each candidate, species-specific ΔlogP scores are computed
as the difference between the mean log-probability of the reference amino acid
assigned by each species-adapted model (Chlamydomonas and rice) and that
assigned by the base model.

**(c) Candidate ranking.** All collected polyQ proteins are ranked by
ΔlogP(*Chlamy*), with species of origin indicated by colour. Human huntingtin
(HTT) ranks prominently among exogenous candidates and is selected as a
canonical, length-sensitive polyQ scaffold for downstream functional
characterisation in *Chlamydomonas*.

---

**Supplementary Figure 3. Deviation of species-adapted ESM2 models from the base
model across polyQ lengths.**

Delta logP(ref) relative to the base ESM2 model for the rice-adapted
and *Chlamydomonas*-adapted variants (3 epochs each), plotted across
polyQ tract lengths Q = 10–72. Raw per-position values (faint lines) are
overlaid with 5-point moving averages (bold lines). Both species-adapted models
exhibit a consistent peak of deviation from the base model within the
intermediate polyQ range Q ≈ 20–32 (shaded), with absolute deviations
diminishing at shorter (Q < 18) and longer (Q > 40) repeat lengths. This
pattern suggests that the Q ≈ 20–32 interval represents a transition zone in
which species-specific proteome context most sensitively modulates sequence
likelihood.

---

## Supplementary Methods

### ESM2 model and species-adapted fine-tuning

The base model is **ESM2-t33-650M-UR50D** (650M parameters), a pretrained
protein language model released by Meta AI<sup>1</sup>. Full proteome sequences
of *Chlamydomonas reinhardtii* (15,277 sequences after filtering) and
*Oryza sativa* (41,787 sequences) were obtained from UniProt reference
proteomes. From the base ESM2 checkpoint, we performed continued masked
language modelling (MLM) on each species' proteome separately, masking 15% of
residues per sequence and training the model to recover the original amino
acids. Each species proteome was used to fine-tune a separate model for
3 epochs.

### Scoring metric and species-adapted ΔlogP

We adopted the pseudo-log-likelihood (PLL) scoring framework from the original
ESM2 study<sup>1</sup>. For a query protein, the logP(ref) at position *i* is
defined as the log-probability assigned to the wild-type amino acid when that
position is masked and the remainder of the sequence is provided as context:

$$
\log P(\mathrm{ref})_i = \log P(\mathrm{aa} = \mathrm{ref}_i \mid \mathbf{x}_{[1,L]\setminus\{i\}})
$$

The sequence-level score, $\overline{\log P(\mathrm{ref})}$, is the arithmetic mean over all
scored positions. A value close to 0 (*P* ≈ 1) indicates strong model
confidence in the reference residue; more negative values reflect greater
uncertainty.

The species-adapted delta score for a candidate protein is defined as:

$$
\Delta\log P_{\mathrm{species}} = \overline{\log P(\mathrm{ref})}_{\mathrm{species\text{-}adapted}} - \overline{\log P(\mathrm{ref})}_{\mathrm{base}}
$$

isolating the effect of species-specific fine-tuning. A positive ΔlogP
suggests that the candidate's sequence composition is more concordant with the
endogenous proteome landscape of that species.

### PolyQ sequence construction and candidate collection

To systematically evaluate polyQ length effects, we constructed a series of
synthetic sequences based on the human *HTT* (huntingtin) exon 1 backbone
(N-terminal flank: 17 residues, MATLEKLMKAFESLKSF; C-terminal proline-rich
region: 51 residues). The polyQ tract length was varied from Q = 10 to Q = 72
in unit increments (63 sequences, total length 78–140 amino acids), spanning
the clinically relevant range from sub-pathogenic to fully penetrant pathogenic
expansions.

Natural polyQ candidate proteins (≥10 consecutive glutamine residues) were
identified from the two proteomes and scored under the base
and species-adapted models. Ranking was performed by
ΔlogP(*Chlamy*), with the rice-adapted model serving as an orthogonal
species-specificity reference.

### Reference

1. Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
   structure with a language model. *Science* **379**, 1123–1130 (2023).

---

---

## 补充图 2

**基于 ESM2 的物种适应性评分识别与衣藻兼容的外源 polyQ 支架蛋白。**

用于优先选择含 polyQ 蛋白进行衣藻功能测试的物种适应性评分策略示意图。

**(a) 模型构建。** 预训练的 ESM2-650M 蛋白质语言模型（基础模型）分别在
衣藻（*C. reinhardtii*）和水稻（*O. sativa*）蛋白质组上进行掩码语言建模微调，
得到衣藻适应性模型和水稻适应性模型。各模型分别在其蛋白质组上进行 3 个 epoch
的继续掩码语言建模训练。

**(b) 候选蛋白收集与评分。** 从衣藻和水稻的蛋白质组中收集含有连续
polyQ 区段（≥10 个谷氨酰胺残基）的蛋白质。对每个候选蛋白，分别计算各物种特异性
ΔlogP 值，即衣藻适应性模型和水稻适应性模型给出的参考氨基酸平均对数概率
与基础模型给出的平均对数概率之差。

**(c) 候选排序。** 所有收集到的 polyQ 蛋白按 ΔlogP(*衣藻*) 排序，颜色标识
物种来源。人源亨廷顿蛋白（HTT）在外源候选蛋白中排名靠前，被选定为经典的长度
敏感型 polyQ 支架蛋白，用于后续衣藻异源表达功能鉴定。

---

**补充图 3. 物种适应性 ESM2 模型相对基础模型的偏差随 polyQ 长度的变化。**

水稻适应性（3 epoch）和衣藻适应性（3 epoch）ESM2 变体在 polyQ 长度
Q = 10–72 范围内相对于基础模型的 Delta logP(ref)。原始逐点值（细线）叠加
5 点滑动平均（粗线）。两个物种适应性模型均在中间 polyQ 区间 Q ≈ 20–32
（阴影区域）呈现一致的偏差峰值，在较短（Q < 18）和较长（Q > 40）重复长度下
绝对偏差减小。该模式提示 Q ≈ 20–32 区间构成一个过渡区，物种特异性蛋白质组
背景在此处对序列似然的调控最为敏感。

---

## 补充方法

### ESM2 模型与分物种微调

基础模型为 Meta AI 发布的预训练蛋白质语言模型 **ESM2-t33-650M-UR50D**
（6.5 亿参数）<sup>1</sup>。从 UniProt 参考蛋白质组获取衣藻
（*Chlamydomonas reinhardtii*，过滤后 15,277 条序列）和水稻
（*Oryza sativa*，41,787 条序列）的全蛋白质组序列。在基础 ESM2 检查点上，
分别对各物种蛋白质组进行继续掩码语言建模（MLM）训练，每次随机掩码序列中
15% 的残基并要求模型恢复原始氨基酸。各物种蛋白质组分别用于微调一个独立模型，
均训练 3 个 epoch。

### 评分指标与物种适应性 ΔlogP

采用 ESM2 原始论文<sup>1</sup> 中的伪对数似然（pseudo-log-likelihood, PLL）
评分框架。对于查询蛋白，位置 *i* 的 logP(ref) 定义为该位置被掩码、其余序列作为
上下文时，模型对野生型氨基酸分配的对数概率：

$$
\log P(\mathrm{ref})_i = \log P(\mathrm{aa} = \mathrm{ref}_i \mid \mathbf{x}_{[1,L]\setminus\{i\}})
$$

序列水平均值 $\overline{\log P(\mathrm{ref})}$ 取所有评分位置的算术平均。值接近 0（*P* ≈ 1）表示
模型对该残基高度认可；负值越大反映不确定性越高。

候选蛋白的物种适应性差异分数定义为：

$$
\Delta\log P_{\text{物种}} = \overline{\log P(\mathrm{ref})}_{\text{物种适应性模型}} - \overline{\log P(\mathrm{ref})}_{\text{基础模型}}
$$

隔离了物种特异性微调对序列评估的影响。ΔlogP 为正提示候选蛋白的序列组成与该
物种内源蛋白质组景观更为吻合。

### PolyQ 序列构建与候选蛋白收集

为系统评估 polyQ 区段长度的影响，基于人类 *HTT*（huntingtin）exon 1 骨架构建
合成序列（N 端侧翼 17 残基 MATLEKLMKAFESLKSF，C 端富脯氨酸区 51 残基）。
polyQ 区段长度从 Q = 10 至 Q = 72 逐位变化（共 63 条序列，总长 78–140 个
氨基酸），覆盖从亚致病到完全外显致病的临床相关范围。

从衣藻和水稻蛋白质组中识别天然含 polyQ 区段（≥10 个连续谷氨酰胺）的
候选蛋白，在基础模型和两个物种适应性模型下评分，按 ΔlogP(*衣藻*) 排序，
水稻适应性模型作为正交的物种特异性参照。

### 参考文献

1. Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
   structure with a language model. *Science* **379**, 1123–1130 (2023).

