# Privacy/leakage checklist (living document)

Check every diff touching listing data against this list. Add a new item
the moment a new leakage vector is discovered — don't wait to "batch" it.

- [ ] No dealer/agent name or contact field reaches a template, API
      response, or export.
- [ ] No raw photo/media URL reaches a template, API response, or export.
- [ ] No phone-number-shaped field reaches a template, API response, or
      export.
- [ ] No new join re-introduces a previously-dropped column from the raw
      source file.
- [ ] All SQL in the diff is parameterized.
- [ ] No write path touches `data/raw/`.
- [ ] Any new derived table/cache has a computation-date + source-version
      header.
