# Beobachtete Ressourcen und konservative Projektion

Prozess-CPU und Wall werden getrennt berichtet. Die äußeren /usr/bin/time-Aufzeichnungen enthalten Start/Import, Hauptprozess und Kindprozesse. Der wissenschaftliche Ledger ist ein konservativer Belastungszähler und enthält auch obere Schranken; er ist keine behauptete exakte Messung sämtlicher CPU-Zeit.

| Phase | Gemessene CPU s | Gemessene Wall s |
|---|---:|---:|
| Erfolgreiches Profiling | 706.070 | 959.270 |
| Versiegelter Preflight | 530.190 | 838.640 |
| Training + 72 Diagnostic-Paare | 3097.140 | 7052.030 |
| Deskriptive Ergebnisaufbereitung | 115.720 | 129.140 |

Engineering + fokussierte Tests: 583.335901 gemessene CPU-Sekunden. Das erste fehlgeschlagene synthetische Profil wurde mit 998.381727 Sekunden konservativ belastet; seine exakte CPU ist unbekannt. Bekannte gemessene CPU einschließlich Ergebnisaufbereitung: 5032.455901 s. Diese Messung ist getrennt von der im Voraus eingefrorenen Pilot-Projektion 41370.753270 s / 2102448583.75 Bytes (11.491876 h / 1.958058 GiB).

Finale Berichts-/Hash-/Git-Kosten sind durch den vorab reservierten 600s-Berichtsblock und die dreifache Hash-Passage abgedeckt; endgültige gespeicherte Bytes und Hash-Readback stehen in final_readback.json und im abschließenden Git-Readback. Kein zusätzlicher Performance-Reset durch die deskriptive Aufbereitung.
