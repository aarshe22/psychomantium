import { ReactNode } from "react";

type Props = {
  id: string;
  title: string;
  open: boolean;
  onToggle: (id: string) => void;
  children: ReactNode;
};

export default function Accordion({ id, title, open, onToggle, children }: Props) {
  return (
    <section className={`acc ${open ? "open" : ""}`}>
      <button type="button" className="acc-h" onClick={() => onToggle(id)} aria-expanded={open}>
        <span>{title}</span>
        <span className="acc-chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="acc-b">{children}</div>}
    </section>
  );
}
