import {
  createContext,
  useContext,
  useLayoutEffect,
  useRef,
  useSyncExternalStore,
  type MutableRefObject,
  type ReactNode,
} from "react";

import { useAppState, type AppState, type AppActions } from "../hooks/useAppState";

type AppContextValue = AppState & AppActions;
type EqualityFn<T> = (left: T, right: T) => boolean;
type AppFunctionActions = {
  [Key in keyof AppActions as AppActions[Key] extends (...args: any[]) => any ? Key : never]: AppActions[Key];
};

interface AppStateStore {
  emit: () => void;
  getState: () => AppState;
  replaceState: (nextState: AppState) => void;
  subscribe: (listener: () => void) => () => void;
}

const AppStateStoreContext = createContext<AppStateStore | null>(null);
const AppActionsContext = createContext<AppFunctionActions | null>(null);

function createAppStateStore(initialState: AppState): AppStateStore {
  let state = initialState;
  const listeners = new Set<() => void>();

  return {
    emit() {
      listeners.forEach((listener) => {
        listener();
      });
    },
    getState() {
      return state;
    },
    replaceState(nextState) {
      state = nextState;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

function splitAppValue(value: AppContextValue): {
  actions: AppFunctionActions;
  state: AppState;
} {
  const actions = {} as AppFunctionActions;
  const state = {} as AppState;

  for (const [rawKey, entry] of Object.entries(value)) {
    const key = rawKey as keyof AppContextValue;
    if (typeof entry === "function") {
      (actions as Record<string, unknown>)[String(key)] = entry;
      continue;
    }
    ((state as unknown) as Record<string, unknown>)[String(key)] = entry;
  }

  return { actions, state };
}

function createStableActionWrappers(
  actionsRef: MutableRefObject<AppFunctionActions>,
): AppFunctionActions {
  const wrapped = {} as AppFunctionActions;

  for (const key of Object.keys(actionsRef.current) as Array<keyof AppFunctionActions>) {
    (wrapped as Record<string, unknown>)[String(key)] = ((...args: unknown[]) => {
      const next = actionsRef.current[key] as (...callArgs: unknown[]) => unknown;
      return next(...args);
    }) as AppFunctionActions[typeof key];
  }

  return wrapped;
}

function useAppStateStore(): AppStateStore {
  const store = useContext(AppStateStoreContext);
  if (!store) {
    throw new Error("useAppSelector must be used within an AppProvider");
  }
  return store;
}

export function shallowEqual<T extends Record<string, unknown>>(left: T, right: T): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  return leftKeys.every((key) => Object.is(left[key], right[key]));
}

export function AppProvider({ children }: { children: ReactNode }) {
  const value = useAppState();
  const { actions, state } = splitAppValue(value);

  const storeRef = useRef<AppStateStore | null>(null);
  if (!storeRef.current) {
    storeRef.current = createAppStateStore(state);
  }

  const pendingNotifyRef = useRef(false);
  if (storeRef.current.getState() !== state) {
    storeRef.current.replaceState(state);
    pendingNotifyRef.current = true;
  }

  useLayoutEffect(() => {
    if (!pendingNotifyRef.current) {
      return;
    }
    pendingNotifyRef.current = false;
    storeRef.current?.emit();
  });

  const actionsRef = useRef(actions);
  actionsRef.current = actions;

  const stableActionsRef = useRef<AppFunctionActions | null>(null);
  if (!stableActionsRef.current) {
    stableActionsRef.current = createStableActionWrappers(actionsRef);
  }

  return (
    <AppStateStoreContext.Provider value={storeRef.current}>
      <AppActionsContext.Provider value={stableActionsRef.current}>
        {children}
      </AppActionsContext.Provider>
    </AppStateStoreContext.Provider>
  );
}

export function useAppSelector<T>(
  selector: (state: AppState) => T,
  isEqual: EqualityFn<T> = Object.is,
): T {
  const store = useAppStateStore();
  const hasSelectionRef = useRef(false);
  const selectionRef = useRef<T | null>(null);

  return useSyncExternalStore(
    store.subscribe,
    () => {
      const nextSelection = selector(store.getState());
      if (hasSelectionRef.current && isEqual(selectionRef.current as T, nextSelection)) {
        return selectionRef.current as T;
      }
      selectionRef.current = nextSelection;
      hasSelectionRef.current = true;
      return nextSelection;
    },
    () => selector(store.getState()),
  );
}

export function useAppActions(): AppFunctionActions {
  const actions = useContext(AppActionsContext);
  if (!actions) {
    throw new Error("useAppActions must be used within an AppProvider");
  }
  return actions;
}

export function useApp(): AppContextValue {
  const state = useAppSelector((currentState) => currentState);
  const actions = useAppActions();
  return {
    ...state,
    ...actions,
  } as AppContextValue;
}
