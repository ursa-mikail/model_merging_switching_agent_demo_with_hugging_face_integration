interface Props {
  message: string;
  onDismiss: () => void;
}

export default function Toast({ message, onDismiss }: Props) {
  return (
    <div className="toast" role="alert">
      <strong>Something went wrong</strong>
      {message}
      <div style={{ marginTop: 10, textAlign: "right" }}>
        <button
          className="icon-btn"
          style={{ width: "auto", padding: "4px 10px", fontSize: 12 }}
          onClick={onDismiss}
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}
