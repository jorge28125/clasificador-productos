# Graph Report - clasificador_de_productos  (2026-11-11)

## Corpus Check
- 54 files · ~491,430 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .pt 5, (none) 4, .cache 2)

## Summary
- 157 nodes · 176 edges · 4 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- preparar_grocery.py
- app.py
- productos/args.yaml
- data.yaml

## God Nodes (most connected - your core abstractions)
1. `main()` - 6 edges
2. `main()` - 6 edges
3. `leer_clases()` - 5 edges
4. `copiar_split()` - 4 edges
5. `construir_texto_voz()` - 3 edges
6. `predict()` - 3 edges
7. `cargar_catalogo()` - 3 edges
8. `names` - 3 edges
9. `index()` - 2 edges
10. `validar_dataset()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (4 total, 0 thin omitted)

### Community 0 - "preparar_grocery.py"
Cohesion: 0.17
Nodes (7): cargar_catalogo(), main(), validar_dataset(), copiar_split(), leer_clases(), main(), traducir()

### Community 1 - "app.py"
Cohesion: 0.18
Nodes (3): construir_texto_voz(), index(), predict()

### Community 2 - "productos/args.yaml"
Cohesion: 0.02
Nodes (114): agnostic_nms, amp, angle, augment, auto_augment, batch, bgr, box (+106 more)

### Community 3 - "data.yaml"
Cohesion: 0.29
Nodes (6): names, 0, 1, path, train, val

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Should `productos/args.yaml` be split into smaller, more focused modules?**
  _Cohesion score 0.017391304347826087 - nodes in this community are weakly interconnected._