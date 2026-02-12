from typing import Iterable, Optional, Tuple, Union


import pandas as pd
import random
import time

try:
    from .common.config import PipelineConfig
    from .common.io_utils import maybe_save_csv
    from .common.time_utils import resolve_time_range
    from .common.transforms import to_unix_timestamp
except ImportError:  # Allow running as a script without package context.
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data_preprocessing.metrics.common.config import PipelineConfig
    from data_preprocessing.metrics.common.io_utils import maybe_save_csv
    from data_preprocessing.metrics.common.time_utils import resolve_time_range
    from data_preprocessing.metrics.common.transforms import to_unix_timestamp


try:
    from pytrends.request import TrendReq
    from pytrends.exceptions import TooManyRequestsError
except Exception:  # pragma: no cover - optional dependency
    TrendReq = None
    TooManyRequestsError = Exception


def _iter_date_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    overlap_days: int,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive.")
    overlap_days = max(0, min(overlap_days, chunk_days - 1))
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        yield current, chunk_end
        if overlap_days > 0:
            next_start = chunk_end - pd.Timedelta(days=overlap_days)
            if next_start <= current:
                next_start = chunk_end
        else:
            next_start = chunk_end
        current = next_start


def _fetch_trend_chunk(
    trend: "TrendReq",
    keyword: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: PipelineConfig,
) -> pd.DataFrame:
    timeframe = f"{start.date()} {end.date()}"
    max_retries = max(0, config.google_trend_max_retries)
    backoff_base = max(0.1, config.google_trend_backoff_seconds)

    for attempt in range(max_retries + 1):
        try:
            trend.build_payload(
                [keyword],
                cat=config.google_trend_category,
                timeframe=timeframe,
                geo=config.google_trend_geo,
            )

            df = trend.interest_over_time()
            if df.empty:
                return df

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            if keyword not in df.columns:
                return pd.DataFrame()

            df = df[[keyword]].rename(columns={keyword: "google_trend"})
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce").floor("D")
            df = df[~df.index.isna()]
            return df
        except TooManyRequestsError as exc:
            if attempt >= max_retries:
                if config.debug:
                    print(f"[DEBUG] google_trend 429 after {attempt} retries: {exc}")
                return pd.DataFrame()
            sleep_seconds = backoff_base * (2**attempt) + random.uniform(0, backoff_base)
            if config.debug:
                print(
                    f"[DEBUG] google_trend 429, retry {attempt + 1}/{max_retries} "
                    f"sleep={sleep_seconds:.1f}s"
                )
            time.sleep(sleep_seconds)
        except Exception as exc:
            if config.debug:
                print(f"[DEBUG] google_trend chunk failed: {exc}")
            return pd.DataFrame()
    return pd.DataFrame()


def _compute_scale(
    reference: pd.DataFrame,
    chunk_df: pd.DataFrame,
) -> Tuple[float, int]:
    overlap_index = reference.index.intersection(chunk_df.index)
    scale = 1.0
    if len(overlap_index) > 0:
        prev = reference.loc[overlap_index, "google_trend"]
        curr = chunk_df.loc[overlap_index, "google_trend"]
        prev_mean = prev[prev > 0].mean()
        curr_mean = curr[curr > 0].mean()
        if pd.notna(prev_mean) and pd.notna(curr_mean) and curr_mean > 0:
            scale = float(prev_mean / curr_mean)
    return scale, len(overlap_index)


def _merge_chunk(
    combined: pd.DataFrame,
    chunk_df: pd.DataFrame,
    direction: str,
    debug: bool,
) -> pd.DataFrame:
    if combined is None or combined.empty:
        return chunk_df.copy()

    scale, overlap_len = _compute_scale(combined, chunk_df)
    if debug:
        print(f"[DEBUG] google_trend chunk scale={scale:.4f} overlap={overlap_len}")

    scaled = chunk_df.copy()
    scaled["google_trend"] = scaled["google_trend"] * scale
    if overlap_len > 0:
        scaled = scaled.loc[~scaled.index.isin(combined.index)]

    if direction == "prepend":
        combined = pd.concat([scaled, combined]).sort_index()
    else:
        combined = pd.concat([combined, scaled]).sort_index()
    return combined


def _stitch_chunks_anchored(
    chunks: Iterable[pd.DataFrame],
    anchor_idx: int,
    debug: bool = False,
) -> pd.DataFrame:
    chunk_list = [c for c in chunks if c is not None and not c.empty]
    if not chunk_list:
        return pd.DataFrame()
    anchor_idx = max(0, min(anchor_idx, len(chunk_list) - 1))

    combined = chunk_list[anchor_idx].copy()
    # stitch earlier chunks (prepend)
    for i in range(anchor_idx - 1, -1, -1):
        combined = _merge_chunk(combined, chunk_list[i], "prepend", debug)
    # stitch later chunks (append)
    for i in range(anchor_idx + 1, len(chunk_list)):
        combined = _merge_chunk(combined, chunk_list[i], "append", debug)
    return combined


def fetch_google_trend(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    if TrendReq is None:
        if config.debug:
            print("[DEBUG] pytrends not installed; google trends skipped")
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "google_trend.csv", enabled=save)
        return output

    trend = TrendReq(hl=config.google_trend_hl, tz=config.google_trend_tz)

    total_days = max(1, (end_ts - start_ts).days)
    chunk_days = max(1, config.google_trend_chunk_days)
    overlap_days = max(0, config.google_trend_overlap_days)
    anchor_days = max(1, config.google_trend_anchor_days)

    if total_days <= chunk_days:
        chunks = [
            _fetch_trend_chunk(
                trend, config.google_trend_keyword, start_ts, end_ts, config
            )
        ]
        anchor_idx = 0
    else:
        chunks = []
        for chunk_start, chunk_end in _iter_date_chunks(
            start_ts, end_ts, chunk_days, overlap_days
        ):
            chunks.append(
                _fetch_trend_chunk(
                    trend,
                    config.google_trend_keyword,
                    chunk_start,
                    chunk_end,
                    config,
                )
            )
        anchor_start = end_ts - pd.Timedelta(days=anchor_days)
        anchor_idx = 0
        for idx, chunk_df in enumerate(chunks):
            if chunk_df is None or chunk_df.empty:
                continue
            if chunk_df.index.min() <= anchor_start <= chunk_df.index.max():
                anchor_idx = idx
        if config.debug:
            print(
                f"[DEBUG] google_trend anchor_idx={anchor_idx} "
                f"anchor_start={anchor_start.date()}"
            )

    stitched = _stitch_chunks_anchored(chunks, anchor_idx, debug=config.debug)
    if stitched.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "google_trend.csv", enabled=save)
        return output

    stitched = stitched.loc[
        (stitched.index >= start_ts.floor("D"))
        & (stitched.index < end_ts.floor("D") + pd.Timedelta(days=1))
    ]
    stitched = stitched.sort_index()
    df = stitched.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    output = df[["timestamp", "google_trend"]].copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "google_trend.csv", enabled=save)
    return output
