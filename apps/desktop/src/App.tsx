import { AppProvider } from "@/context/AppContext";
import AppShell from "@/components/AppShell";
import "./styles.css";

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}
