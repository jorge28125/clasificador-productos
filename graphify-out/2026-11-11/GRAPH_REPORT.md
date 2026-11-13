# Graph Report - clasificador_de_productos  (2026-11-11)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 34 nodes · 57 edges · 7 communities (2 shown, 5 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- preparar_grocery.py
- entrenar.py
- app.py
- leer_clases
- construir_texto_voz
- predict

## God Nodes (most connected - your core abstractions)
1. `main()` - 6 edges
2. `main()` - 6 edges
3. `leer_clases()` - 5 edges
4. `copiar_split()` - 4 edges
5. `construir_texto_voz()` - 4 edges
6. `cargar_catalogo()` - 3 edges
7. `predict()` - 3 edges
8. `traducir()` - 2 edges
9. `validar_dataset()` - 2 edges
10. `cargar_catalogo()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `copiar_split()` --calls--> `shutil`  [EXTRACTED]
  preparar_grocery.py →   _Bridges community 0 → community 1_
- `main()` --calls--> `json`  [EXTRACTED]
  preparar_grocery.py →   _Bridges community 0 → community 2_
- `main()` --calls--> `leer_clases()`  [EXTRACTED]
  preparar_grocery.py → preparar_grocery.py  _Bridges community 0 → community 3_
- `cargar_catalogo()` --calls--> `json`  [EXTRACTED]
  entrenar.py →   _Bridges community 1 → community 2_
- `predict()` --calls--> `construir_texto_voz()`  [EXTRACTED]
  app.py → app.py  _Bridges community 5 → community 6_

## Import Cycles
- None detected.

## Communities (7 total, 5 thin omitted)

### Community 0 - "preparar_grocery.py"
Cohesion: 0.48
Nodes (3): copiar_split(), main(), traducir()

### Community 1 - "entrenar.py"
Cohesion: 0.48
Nodes (3): cargar_catalogo(), main(), validar_dataset()

## Knowledge Gaps
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `leer_clases()` connect `leer_clases` to `preparar_grocery.py`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `main()` connect `preparar_grocery.py` to `app.py`, `leer_clases`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `construir_texto_voz()` connect `construir_texto_voz` to `app.py`, `predict`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._