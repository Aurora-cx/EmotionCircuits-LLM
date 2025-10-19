#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/06_emotion_circuit_integration/5_integrate_global_circuit.py
"""
按子层重要性为每个情绪组合全局电路配额（神经元+注意力头），并输出每层选中的组件 ID
Integrate global circuit quota (neurons + attention heads) per emotion based on sublayer importance

核心逻辑 Core Logic:
  1. 读取子层重要性表（I_final），将负值截断为 0，得到权重 w
     Read sublayer importance table (I_final), clip negative values to 0 to get weight w
  2. 先做"最小保障"（mlp≥min_mlp, attn≥min_attn），再将剩余配额按权重分配
     First ensure minimum guarantee (mlp≥min_mlp, attn≥min_attn), then allocate remaining quota by weight
  3. 单子层上限 cap，防止过度集中；缺口回流同 kind 的高权重子层
     Cap per sublayer to prevent over-concentration; deficit flows back to high-weight sublayers of same kind
  4. 从 per-layer 排名表里按层内配额 k 取前 k 个组件（不足则回流/记录）
     Pick top k components from per-layer ranking table (backfill/log if insufficient)
  5. 多情绪循环输出 JSON + 分配日志 CSV，便于后续注入实验读取
     Output JSON + allocation log CSV for each emotion for subsequent injection experiments

输入输出 Input/Output:
  - 输入 Input:
    - outputs/{model_name}/06_emotion_circuit_integration/importance_all_alpha_mean_emotion_all_probability.csv
    - outputs/{model_name}/04_local_components_identification/mlp_neurons/contrib_mean_{emotion}.csv
    - outputs/{model_name}/04_local_components_identification/attention_heads/head_importance_{emotion}.csv
  - 输出 Output:
    - outputs/{model_name}/06_emotion_circuit_integration/global_circuit/{emotion}.json
    - outputs/{model_name}/06_emotion_circuit_integration/global_circuit/{emotion}_alloc_log.csv
"""

import os, json, argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# ============== 路径与基本配置 ==============
# ============== Paths and Basic Configuration ==============
# 项目根目录：自动获取脚本所在位置的上两级目录
# Project root: automatically get the directory two levels up from the script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME = "llama32_3b"
N_LAYERS = 28
HEADS_PER_LAYER = 24  # 3B: 3072/24=128

# ============== 工具函数 ==============
# ============== Utility Functions ==============
def ensure_dir(p):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(p, exist_ok=True)
    return p

def seed_all(seed:int):
    """
    设置随机种子
    Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)

# ============== 数据加载 ==============
# ============== Data Loading ==============
def load_importance_table(path_csv:str, emotion:str):
    """
    加载子层重要性表
    Load sublayer importance table
    
    期望列 Expected columns: emotion, kind, layer, I_final
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Importance file not found: {path_csv}")
    df = pd.read_csv(path_csv)
    
    # 期望列：emotion, kind, layer, I_final
    # Expected columns: emotion, kind, layer, I_final
    need = {"emotion","kind","layer","I_final"}
    if not need.issubset(df.columns):
        raise ValueError(f"Importance csv缺列，需至少包含 Missing columns, need at least {need}，实际列 actual columns={list(df.columns)}")
    
    df = df[df["emotion"]==emotion].copy()
    
    # 保留合法 kind/layer
    # Keep valid kind/layer
    df = df[df["kind"].isin(["mlp","attention"])]
    df = df[(df["layer"]>=0) & (df["layer"]<N_LAYERS)]
    
    # 负值截断为 0
    # Clip negative values to 0
    df["w"] = np.clip(df["I_final"].astype(float).values, a_min=0.0, a_max=None)
    
    return df.reset_index(drop=True)

def load_neuron_rank_table(neuron_dir:str, emotion:str):
    """
    加载神经元排名表
    Load neuron ranking table
    
    返回 Returns: {layer -> [neuron_id...]}
    """
    p = os.path.join(neuron_dir, f"contrib_mean_{emotion}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Neuron rank csv not found: {p}")
    df = pd.read_csv(p)
    
    need = {"layer","neuron","C_mean"}
    if not need.issubset(df.columns):
        raise ValueError(f"Neuron rank csv需列 need columns {need}，实际列 actual columns={list(df.columns)}")
    
    # 分层排序（降序：贡献越大越前）
    # Sort by layer (descending: larger contribution first)
    df = df.sort_values(["layer","C_mean"], ascending=[True, False]).reset_index(drop=True)
    
    # {layer -> [neuron_id...]}
    out = {L: g["neuron"].astype(int).to_list() for L, g in df.groupby("layer")}
    
    # 缺层补空
    # Fill missing layers with empty list
    for L in range(N_LAYERS):
        out.setdefault(L, [])
    
    return out

def load_head_rank_table(path_csv:str):
    """
    加载注意力头排名表
    Load attention head ranking table
    
    返回 Returns: {layer -> [head_id...]}
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Head rank csv not found: {path_csv}")
    df = pd.read_csv(path_csv)
    
    # 兼容不同的列名：score 或 mean_importance
    # Compatible with different column names: score or mean_importance
    if "mean_importance" in df.columns:
        score_col = "mean_importance"
    elif "score" in df.columns:
        score_col = "score"
    else:
        raise ValueError(f"Head rank csv需列 need column 'score' or 'mean_importance'，实际列 actual columns={list(df.columns)}")
    
    need = {"layer","head"}
    if not need.issubset(df.columns):
        raise ValueError(f"Head rank csv需列 need columns {need}，实际列 actual columns={list(df.columns)}")
    
    # 分层排序（降序）
    # Sort by layer (descending)
    df = df.sort_values(["layer", score_col], ascending=[True, False]).reset_index(drop=True)
    
    # layer 上限 24 个 head
    # Cap at 24 heads per layer
    out = {L: g["head"].astype(int).to_list()[:HEADS_PER_LAYER] for L, g in df.groupby("layer")}
    
    for L in range(N_LAYERS):
        out.setdefault(L, [])
    
    return out

# ============== 分配算法 ==============
# ============== Allocation Algorithms ==============
def floor_cap_and_largest_remainder(raw, cap_each, total_target):
    """
    向下取整 + 上限截断 + 最大剩余法
    Floor + cap + largest remainder method
    
    参数 Args:
        raw: np.array of float (各子层欲分配的"比例份额" each sublayer's proportional share)
        cap_each: 单子层上限（同一常数或同长数组）cap per sublayer (constant or array)
        total_target: 本轮要分配的总名额（整数）total quota to allocate (integer)
    
    返回 Returns: k（int数组 int array），满足 sum(k)==total_target，且 k[i] ≤ cap_each[i]
    """
    n = len(raw)
    if isinstance(cap_each, (int, float)):
        cap = np.full(n, int(cap_each), dtype=int)
    else:
        cap = np.asarray(cap_each, dtype=int)

    # 先 floor
    # First floor
    k = np.floor(raw).astype(int)
    
    # 上限截断
    # Cap truncation
    k = np.minimum(k, cap)
    
    # 剩余
    # Remainder
    remain = int(total_target - k.sum())
    if remain <= 0:
        return k

    # 用最大剩余法补齐，同时不越过 cap
    # Use largest remainder method to fill, without exceeding cap
    frac = raw - np.floor(raw)
    
    # 按小数部分从大到小
    # Sort by fractional part descending
    order = np.argsort(-frac)
    for idx in order:
        if remain <= 0:
            break
        if k[idx] < cap[idx]:
            k[idx] += 1
            remain -= 1
    
    # 如果还剩，继续一轮（极少见，通常因为全触顶）
    # If still remaining, continue one more round (rare, usually because all hit cap)
    if remain > 0:
        for idx in order:
            if remain <= 0:
                break
            if k[idx] < cap[idx]:
                k[idx] += 1
                remain -= 1
    
    return k

def allocate_per_kind(df_kind: pd.DataFrame, K_kind:int, min_per_layer:int, cap_ratio:float):
    """
    对单个 kind（mlp 或 attention）做配额分配
    Allocate quota for a single kind (mlp or attention)
    
    参数 Args:
        df_kind: 只包含一个 kind 的表 DataFrame for one kind, 列 columns: ['layer','w']
        K_kind: 该 kind 的总配额（整数）total quota for this kind (integer)
        min_per_layer: 最小保障 minimum guarantee per layer
        cap_ratio: 单子层对该 kind 的上限占比（相对 K_kind）cap ratio per sublayer (relative to K_kind), 例如 e.g. 0.4
    
    返回 Returns: dict {layer -> k_layer}，以及日志 DataFrame and log DataFrame
    """
    layers = np.array(sorted(df_kind["layer"].unique().tolist()), dtype=int)
    # 合并同层（通常就一行）
    # Merge same layer (usually just one row)
    w = np.array([df_kind[df_kind["layer"]==L]["w"].sum() for L in layers], dtype=float)
    
    # 归一
    # Normalize
    if w.sum() <= 1e-12:
        w_norm = np.ones_like(w) / len(w)
    else:
        w_norm = w / w.sum()

    # Step A: 最小保障
    # Step A: Minimum guarantee
    base = np.full_like(w, fill_value=int(min_per_layer), dtype=int)
    K_min = int(base.sum())
    if K_kind < K_min:
        # 降级策略：将 min_per_layer 降为 1（仍不够就 0）
        # Downgrade strategy: reduce min_per_layer to 1 (or 0 if still insufficient)
        if K_kind >= len(layers):
            base = np.ones_like(w, dtype=int)  # 每层至少 1 / at least 1 per layer
            K_min = int(base.sum())
        else:
            base = np.zeros_like(w, dtype=int) # 不做最小保障 / no minimum guarantee
            K_min = 0
    K_rem = int(K_kind - K_min)

    # Step B: 比例分配 + 上限
    # Step B: Proportional allocation + cap
    cap_each = int(math.floor(cap_ratio * K_kind))
    cap_arr = np.full_like(w, cap_each, dtype=int)

    # 比例份额（剩余配额）
    # Proportional share (remaining quota)
    raw = w_norm * max(K_rem, 0)
    add = floor_cap_and_largest_remainder(raw, cap_arr, total_target=max(K_rem, 0))
    k = base + add

    # 日志
    # Log
    log = pd.DataFrame(dict(layer=layers, w=w, w_norm=w_norm, k_assigned=k, k_base=base, k_add=add))
    
    return {int(L): int(kk) for L, kk in zip(layers, k)}, log

def backfill_with_availability(k_plan:dict, ranking:dict, max_head_per_layer:int=None):
    """
    根据层内可用候选数量，校正不可达的配额，并把缺口回流给同 kind 的高权重层
    Correct unreachable quotas based on available candidates per layer, backfill deficit to high-weight layers of same kind
    
    参数 Args:
        k_plan: {layer -> k} 计划配额 planned quota
        ranking: {layer -> [id,...]} 排名表 ranking table
        max_head_per_layer: 层内最大head数（可选）max heads per layer (optional)
    
    返回 Returns: k_final, 缺口分配日志 deficit allocation log
    """
    # 第一轮：按可用数截断
    # First round: truncate by availability
    available = {L: len(ranking.get(L, [])) for L in k_plan}
    cap = {}
    for L in k_plan:
        if max_head_per_layer is not None:
            cap[L] = min(available[L], max_head_per_layer)
        else:
            cap[L] = available[L]
    
    k_used = {L: min(k_plan[L], cap[L]) for L in k_plan}
    deficit = sum(max(0, k_plan[L] - k_used[L]) for L in k_plan)
    surplus_rooms = {L: max(0, cap[L] - k_used[L]) for L in k_plan}

    # 第二轮：把缺口回流到"尚有余量"的层（按计划配额大的优先）
    # Second round: backfill deficit to layers with surplus (prioritize by planned quota)
    if deficit > 0:
        # 计划配额大的先补
        # Prioritize layers with larger planned quota
        order = sorted(k_plan.items(), key=lambda x: (-x[1], x[0]))
        for L, _ in order:
            if deficit <= 0:
                break
            can_add = surplus_rooms.get(L, 0)
            if can_add <= 0:
                continue
            add = min(can_add, deficit)
            k_used[L] += add
            surplus_rooms[L] -= add
            deficit -= add

    # 若仍有缺口，记录但不再处理
    # If deficit still remains, log but don't process further
    final_deficit = deficit

    return k_used, dict(final_deficit=final_deficit, cap=cap, available=available)

def pick_top_ids(ranking:dict, k_used:dict):
    """
    从排名表中选取前 k 个 ID
    Pick top k IDs from ranking table
    
    参数 Args:
        ranking: {layer -> [id,...]}
        k_used:  {layer -> k}
    
    返回 Returns: {layer -> [ids]}
    """
    sel = {}
    for L, k in k_used.items():
        cand = ranking.get(L, [])
        sel[L] = cand[:k] if k>0 else []
    return sel

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    """
    主函数
    Main function
    """
    ap = argparse.ArgumentParser(description="按子层重要性分配全局电路配额 Allocate global circuit quota by sublayer importance")
    ap.add_argument("--model_name", default=MODEL_NAME,
                    help="模型名称 Model name")
    ap.add_argument("--emotions", default="anger,sadness,happiness,fear,disgust,surprise",
                    help="情绪列表 Emotion list")
    ap.add_argument("--K_total", type=int, default=560,
                    help="全局总配额（神经元+head）Global total quota (neurons + heads)")
    ap.add_argument("--ratio_mode", choices=["auto","fixed"], default="fixed",
                    help="配额比例模式 Quota ratio mode: auto (by weight) or fixed")
    ap.add_argument("--ratio_fixed", default="0.7,0.3",
                    help="fixed 模式下的 (MLP:Attn) 比例 Fixed mode (MLP:Attn) ratio, 例如 e.g. 0.7,0.3")
    ap.add_argument("--min_mlp", type=int, default=4,
                    help="MLP 最小保障 MLP minimum guarantee per layer")
    ap.add_argument("--min_attn", type=int, default=2,
                    help="Attention 最小保障 Attention minimum guarantee per layer")
    ap.add_argument("--cap_ratio", type=float, default=0.4,
                    help="单子层上限占比（相对该 kind 的 K）Cap ratio per sublayer (relative to K for that kind)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子 Random seed")

    args = ap.parse_args()
    seed_all(args.seed)

    # 路径配置
    # Path configuration
    base_dir = PROJECT_ROOT / "outputs" / args.model_name
    dir_06 = base_dir / "06_emotion_circuit_integration"
    dir_04 = base_dir / "04_local_components_identification"
    
    importance_csv = dir_06 / "importance_all_alpha_mean_emotion_all_probability.csv"
    neuron_dir = dir_04 / "mlp_neurons"
    head_dir = dir_04 / "attention_heads"
    out_dir = ensure_dir(dir_06 / "global_circuit")

    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    
    # 解析固定比例
    # Parse fixed ratio
    if args.ratio_mode == "fixed":
        try:
            r_mlp, r_attn = [float(x) for x in args.ratio_fixed.split(",")]
        except:
            raise ValueError("--ratio_fixed 解析失败 parsing failed，示例 example：0.7,0.3")
        if r_mlp + r_attn <= 0:
            raise ValueError("ratio sum must be > 0")
        r_sum = r_mlp + r_attn
        r_mlp /= r_sum
        r_attn /= r_sum

    print(f"\n{'='*70}")
    print(f"全局电路配额分配 Global Circuit Quota Allocation")
    print(f"{'='*70}")
    print(f"[配置 CONFIG]")
    print(f"  - K_total={args.K_total}")
    print(f"  - ratio_mode={args.ratio_mode}")
    print(f"  - min_mlp={args.min_mlp}, min_attn={args.min_attn}")
    print(f"  - cap_ratio={args.cap_ratio}")
    print(f"\n[路径 PATHS]")
    print(f"  - importance_csv: {importance_csv}")
    print(f"  - neuron_dir: {neuron_dir}")
    print(f"  - head_dir: {head_dir}")
    print(f"  - output_dir: {out_dir}")
    
    # 检查输入文件
    # Check input files
    if not importance_csv.exists():
        print(f"\n[-] 错误 Error: 重要性文件不存在 Importance file not found: {importance_csv}")
        print(f"    请先运行 Please run first: 4_analyze_sublayer_importance.py")
        return

    for emo in emotions:
        print(f"\n{'='*70}")
        print(f"情绪 Emotion: {emo}")
        print(f"{'='*70}")
        
        # 1) 载入重要性
        # 1) Load importance
        imp = load_importance_table(str(importance_csv), emo)
        
        # 拆成两类
        # Split into two kinds
        imp_mlp = imp[imp["kind"]=="mlp"].copy()
        imp_attn = imp[imp["kind"]=="attention"].copy()

        # 2) 计算 K_mlp / K_attn
        # 2) Calculate K_mlp / K_attn
        if args.ratio_mode == "auto":
            W_mlp = imp_mlp["w"].sum()
            W_attn = imp_attn["w"].sum()
            if W_mlp + W_attn <= 1e-12:
                # 无有效权重，均分
                # No valid weight, split equally
                K_mlp = args.K_total // 2
                K_attn = args.K_total - K_mlp
            else:
                K_mlp = int(round(args.K_total * (W_mlp / (W_mlp + W_attn))))
                K_attn = int(args.K_total - K_mlp)
        else:
            K_mlp = int(round(args.K_total * r_mlp))
            K_attn = int(args.K_total - K_mlp)
        
        print(f"  -> K_mlp={K_mlp}, K_attn={K_attn}")

        # 3) 读取榜单
        # 3) Load ranking tables
        neuron_ranking = load_neuron_rank_table(str(neuron_dir), emo)
        head_csv = head_dir / f"head_importance_{emo}.csv"
        head_ranking = load_head_rank_table(str(head_csv))

        # 4) 分配（各自 kind 内）
        # 4) Allocate (within each kind)
        k_mlp_plan, log_mlp = allocate_per_kind(imp_mlp, K_mlp, args.min_mlp, args.cap_ratio)
        k_attn_plan, log_attn = allocate_per_kind(imp_attn, K_attn, args.min_attn, args.cap_ratio)

        # 5) 考虑层内可用性做回流（神经元不设额外上限；head 每层 ≤ 24）
        # 5) Backfill considering layer availability (no extra cap for neurons; heads ≤ 24 per layer)
        k_mlp_used, info_mlp = backfill_with_availability(k_mlp_plan, neuron_ranking, max_head_per_layer=None)
        k_attn_used, info_attn = backfill_with_availability(k_attn_plan, head_ranking, max_head_per_layer=HEADS_PER_LAYER)

        # 6) 选取具体 ID
        # 6) Pick specific IDs
        sel_mlp = pick_top_ids(neuron_ranking, k_mlp_used)
        sel_attn = pick_top_ids(head_ranking, k_attn_used)

        # 7) 保存 JSON
        # 7) Save JSON
        js = dict(
            emotion=emo,
            K_total=args.K_total,
            ratio_mode=args.ratio_mode,
            ratio_fixed=(args.ratio_fixed if args.ratio_mode=="fixed" else None),
            K_alloc=dict(mlp=K_mlp, attention=K_attn),
            min_per_layer=dict(mlp=args.min_mlp, attention=args.min_attn),
            cap_ratio=args.cap_ratio,
            summary=dict(
                mlp=dict(planned=sum(k_mlp_plan.values()), used=sum(k_mlp_used.values()),
                         final_deficit=info_mlp.get("final_deficit",0)),
                attention=dict(planned=sum(k_attn_plan.values()), used=sum(k_attn_used.values()),
                               final_deficit=info_attn.get("final_deficit",0)),
            ),
            layers=[]
        )
        
        for L in range(N_LAYERS):
            js["layers"].append(dict(
                layer=L,
                mlp=dict(k=k_mlp_used.get(L,0), neurons=sel_mlp.get(L,[])),
                attention=dict(k=k_attn_used.get(L,0), heads=sel_attn.get(L,[]))
            ))

        out_json = out_dir / f"{emo}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)
        print(f"  -> 已保存 Saved: {out_json}")

        # 8) 保存分配日志 CSV（便于画图/论文表）
        # 8) Save allocation log CSV (for plotting/paper tables)
        log_mlp["kind"] = "mlp"
        log_attn["kind"] = "attention"
        log_all = pd.concat([log_mlp, log_attn], ignore_index=True)
        
        # 附带可用/上限/最终使用
        # Include available/cap/final used
        rows = []
        for kind, log_df, info in [("mlp", log_mlp, info_mlp), ("attention", log_attn, info_attn)]:
            for _, r in log_df.iterrows():
                L = int(r["layer"])
                rows.append(dict(
                    emotion=emo, kind=kind, layer=L,
                    w=float(r["w"]), w_norm=float(r["w_norm"]),
                    k_base=int(r["k_base"]), k_add=int(r["k_add"]),
                    k_assigned=int(r["k_assigned"]),
                    available=int(info["available"].get(L, 0)),
                    cap=int(info["cap"].get(L, 0)),
                    k_used=(len(sel_mlp[L]) if kind=="mlp" else len(sel_attn[L]))
                ))
        
        df_log = pd.DataFrame(rows).sort_values(["kind","layer"]).reset_index(drop=True)
        out_log = out_dir / f"{emo}_alloc_log.csv"
        df_log.to_csv(out_log, index=False)
        print(f"  -> 已保存 Saved: {out_log}")
        
        # 显示统计摘要
        # Display statistical summary
        print(f"\n  [统计摘要 Summary]")
        print(f"    MLP: 计划 planned={sum(k_mlp_plan.values())}, 实际 used={sum(k_mlp_used.values())}, 缺口 deficit={info_mlp.get('final_deficit',0)}")
        print(f"    Attention: 计划 planned={sum(k_attn_plan.values())}, 实际 used={sum(k_attn_used.values())}, 缺口 deficit={info_attn.get('final_deficit',0)}")

    print(f"\n{'='*70}")
    print(f"[完成] 全局电路配额分配完成 [Done] Global circuit allocation completed")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
