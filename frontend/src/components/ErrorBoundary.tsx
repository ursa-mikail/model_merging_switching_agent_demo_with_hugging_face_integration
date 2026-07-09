import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches any render-time exception in the tree below it and shows a
 * recoverable error card instead of letting React unmount everything (which
 * would otherwise show as a blank white page with no explanation).
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("Splice UI crashed:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="app">
          <div className="disclosure" style={{ borderLeftColor: "var(--accent-danger)" }}>
            <strong>Something went wrong rendering the UI.</strong>
            <br />
            {this.state.error.message}
            <br />
            <br />
            <button
              className="icon-btn"
              style={{ width: "auto", padding: "8px 14px" }}
              onClick={() => {
                this.setState({ error: null });
                window.location.reload();
              }}
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
