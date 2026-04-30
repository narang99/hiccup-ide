# Claude Code Commands

## Linting
Use the npm script instead of running eslint directly:
```bash
npm run lint
```

This now includes both ESLint checking and TypeScript compilation checking (`tsc -b`).

If you only want ESLint checking without TypeScript:
```bash
npm run lint:eslint
```

For only TypeScript checking:
```bash
npm run typecheck
```

Do not use `eslint .` directly - always use the npm script which may have additional configuration or flags.