import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

function findOffsetIndex(
  starts: number[],
  sizes: number[],
  offset: number,
): number {
  if (starts.length === 0) {
    return 0;
  }
  let low = 0;
  let high = starts.length - 1;
  let answer = 0;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const start = starts[mid] || 0;
    const end = start + (sizes[mid] || 0);
    if (offset < start) {
      high = mid - 1;
    } else if (offset >= end) {
      answer = Math.min(mid + 1, starts.length - 1);
      low = mid + 1;
    } else {
      return mid;
    }
  }
  return answer;
}

export function useVirtualizedList<T extends { id: string }>(args: {
  items: T[];
  containerRef: React.RefObject<HTMLElement | null>;
  listRef: React.RefObject<HTMLElement | null>;
  estimateSize: (item: T) => number;
  overscan?: number;
}) {
  const {
    items,
    containerRef,
    listRef,
    estimateSize,
    overscan = 480,
  } = args;
  const heightsRef = useRef(new Map<string, number>());
  const [measureVersion, setMeasureVersion] = useState(0);
  const [scrollState, setScrollState] = useState({
    scrollTop: 0,
    viewportHeight: 0,
    listOffsetTop: 0,
  });

  const reportSize = useCallback((id: string, size: number) => {
    if (!Number.isFinite(size) || size <= 0) {
      return;
    }
    if (heightsRef.current.get(id) === size) {
      return;
    }
    heightsRef.current.set(id, size);
    setMeasureVersion((current) => current + 1);
  }, []);

  useEffect(() => {
    const validIds = new Set(items.map((item) => item.id));
    let changed = false;
    for (const id of heightsRef.current.keys()) {
      if (!validIds.has(id)) {
        heightsRef.current.delete(id);
        changed = true;
      }
    }
    if (changed) {
      setMeasureVersion((current) => current + 1);
    }
  }, [items]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const update = () => {
      const listOffsetTop = listRef.current?.offsetTop ?? 0;
      setScrollState({
        scrollTop: container.scrollTop,
        viewportHeight: container.clientHeight,
        listOffsetTop,
      });
    };

    update();
    container.addEventListener("scroll", update, { passive: true });

    if (typeof ResizeObserver === "function") {
      const observer = new ResizeObserver(update);
      observer.observe(container);
      if (listRef.current) {
        observer.observe(listRef.current);
      }
      return () => {
        container.removeEventListener("scroll", update);
        observer.disconnect();
      };
    }

    window.addEventListener("resize", update);
    return () => {
      container.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, [containerRef, listRef, items.length]);

  const metrics = useMemo(() => {
    const starts: number[] = new Array(items.length);
    const sizes: number[] = new Array(items.length);
    let totalHeight = 0;
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index]!;
      starts[index] = totalHeight;
      const size = heightsRef.current.get(item.id) ?? estimateSize(item);
      sizes[index] = size;
      totalHeight += size;
    }
    return {
      starts,
      sizes,
      totalHeight,
    };
  }, [estimateSize, items, measureVersion]);

  const relativeScrollTop = Math.max(0, scrollState.scrollTop - scrollState.listOffsetTop);
  const viewportBottom = relativeScrollTop + scrollState.viewportHeight;
  const startIndex = Math.max(
    0,
    findOffsetIndex(metrics.starts, metrics.sizes, Math.max(0, relativeScrollTop - overscan)),
  );
  const endIndex = Math.min(
    items.length - 1,
    findOffsetIndex(metrics.starts, metrics.sizes, viewportBottom + overscan),
  );

  const virtualItems = useMemo(() => {
    if (items.length === 0) {
      return [];
    }
    const next = [];
    for (let index = startIndex; index <= endIndex; index += 1) {
      const item = items[index];
      if (!item) continue;
      next.push({
        item,
        index,
        start: metrics.starts[index] || 0,
        size: metrics.sizes[index] || 0,
      });
    }
    return next;
  }, [endIndex, items, metrics.sizes, metrics.starts, startIndex]);

  return {
    totalHeight: metrics.totalHeight,
    virtualItems,
    reportSize,
  };
}
