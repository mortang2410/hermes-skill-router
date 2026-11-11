/**
 * Skills routed — the Hermes Desktop half of the display (module M13, contract C9).
 *
 * A runtime Hermes desktop plugin: HERMES copies this folder into
 * `$HERMES_HOME/desktop-plugins/hermes-skill-router/` when the plugin lands in a plugins root, the
 * renderer evaluates the file as plain ESM, and `register(ctx)` contributes one pane. The pane is
 * the desktop rail on the right, 260 px wide, collapsed by default, and it lists the skill names
 * the router suggested on the last turn — never a kept row, never text from a request. Under the
 * heading sits one muted caption naming the turn and its local clock time, built from the record's
 * own `turn_id` and `at`; when the record says a model search produced the names, the caption
 * paints the model-search marker instead, because a lookup belongs to no turn.
 *
 * The data comes from the plugin's own backend route, `GET /suggestions` (dashboard/plugin_api.py,
 * mounted at `/api/plugins/hermes-skill-router/`), through `ctx.rest`, so the desktop half never
 * reads the record file itself and never holds a path. It refreshes on mount (the query's first
 * fetch) and whenever the backend broadcasts `plugin.hermes-skill-router.suggestions.updated` after
 * a record write. A 404 means the route is not mounted — the Python backend is not enabled in the
 * profile's `plugins.enabled` list — and the pane paints a plain enable line instead of an error;
 * every other failure paints one short line, so a broken read never throws inside the app.
 *
 * It stays inside the `@hermes/plugin-sdk` surface the admission lint allows: static imports of
 * the SDK and `react/jsx-runtime` only, no prototype patching, no dynamic code, no remote imports,
 * and no reach into the app's DOM.
 */

import { atom, DisclosureCaret, PANES_AREA, queryClient, useQuery, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

/** Plugin id: the contribution namespace and the backend route's plugin namespace. */
export const PLUGIN_ID = 'hermes-skill-router'

/** The one pane contribution's local id and its title. */
export const PANE_ID = 'skills-routed'
export const PANE_TITLE = 'Skills routed'

/** The backend route the pane reads, relative to `/api/plugins/<id>`. */
export const ROUTE = '/suggestions'

/** Broadcast by the rewrite after it writes the record, with the turn id. */
export const UPDATED_EVENT = 'plugin.hermes-skill-router.suggestions.updated'

/** One query key, so the event's invalidation and the pane's query cannot drift apart. */
export const SUGGESTIONS_QUERY_KEY = ['hermes-skill-router', 'suggestions']

/** The pane's fixed rail width, matching `data.width` below. */
export const PANE_WIDTH = '260px'

/** Painted when the backend route is not mounted: the Python backend is disabled. */
export const ENABLE_LINE = 'Enable the plugin in plugins.enabled to serve the suggestions route.'

/** Painted for any route failure that is not a missing route. */
export const READ_ERROR_LINE = 'The suggestions route could not be read.'

/** The record states, names only. */
const IDLE_LINE = 'No suggestions recorded yet.'
const INERT_LINE = 'Router inactive: no suggestions this turn.'
const EMPTY_LINE = 'The last routed turn suggested nothing.'

/** Expanded/collapsed state of the names list; collapsed is the default. */
const $expanded = atom(false)

/**
 * Whether *error* means the backend route is not mounted (HTTP 404).
 *
 * `ctx.rest` rejects with an `Error` carrying `statusCode` for a local backend, and the desktop's
 * IPC bridge re-throws the expected-404 sentinel as `new Error("404: <body>")` with the property
 * dropped, so both shapes are checked. Anything else (500, a dead backend) is a read error.
 */
export function isMissingRoute(error) {
  if (!error) {
    return false
  }

  const status = Number(error.statusCode)

  if (Number.isInteger(status) && status === 404) {
    return true
  }

  return /^\s*(?:Error:\s*)?404\b/.test(String(error.message ?? error))
}

/** The marker a search record paints where a routed turn paints its turn id. */
export const MODEL_SEARCH_LABEL = 'model search'

/**
 * The local clock time of an ISO 8601 record timestamp, `HH:MM`, or `""` when it does not parse.
 *
 * The record stores UTC; the pane shows the reader's own clock, because the caption answers "when
 * did this decision happen for me", not "what did the writer's clock say".
 */
export function shortTime(value) {
  if (typeof value !== 'string' || value.trim() === '') {
    return ''
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}

/**
 * The one caption line under the heading: which decision the names came from and when.
 *
 * Built from the record's own `origin`, `turn_id` and `at` only — never from a message, a prompt or
 * any other field — and empty for `inert` and for an absent record, so a stale turn id can never
 * sit under an inactive or idle state. A `search` record paints the model-search marker instead of
 * a turn id: `model search · <local HH:MM>`. A `turn` record keeps `turn <id> · <local HH:MM>`,
 * dropping either half when it is missing.
 */
export function captionFor(record) {
  if (!record || typeof record !== 'object' || record.status !== 'routed') {
    return ''
  }

  if (record.origin === 'search') {
    return [MODEL_SEARCH_LABEL, shortTime(record.at)].filter(Boolean).join(' · ')
  }

  const turn = typeof record.turn_id === 'string' ? record.turn_id.trim() : ''
  const stamp = shortTime(record.at)

  return [turn ? `turn ${turn}` : '', stamp].filter(Boolean).join(' · ')
}

/**
 * The labels one pane render needs, derived from a route body alone.
 *
 * Names only: `names` is the record's own list (already capped by `maxDisplayNames` when the
 * writer stored it) and every other key of the body is ignored, so a request message, a system
 * prompt or any other text in the record can never reach the pane. A missing record, a corrupt
 * one and an idle answer all collapse to the idle line.
 */
export function summary(record) {
  const status =
    record && typeof record === 'object' && typeof record.status === 'string' ? record.status : 'idle'
  const names =
    status === 'routed' && Array.isArray(record.names)
      ? record.names.filter(name => typeof name === 'string' && name.trim() !== '')
      : []
  const heading = `${PANE_TITLE} (${names.length})`
  const caption = captionFor(record)

  if (status === 'inert') {
    return {
      caption,
      count: 0,
      heading: `${PANE_TITLE} (0)`,
      names: [],
      note: INERT_LINE,
      status: 'inert'
    }
  }

  if (status !== 'routed') {
    return {
      caption,
      count: 0,
      heading: `${PANE_TITLE} (0)`,
      names: [],
      note: IDLE_LINE,
      status: 'idle'
    }
  }

  return {
    caption,
    count: names.length,
    heading,
    names,
    note: names.length ? '' : EMPTY_LINE,
    status: 'routed'
  }
}

export default {
  id: PLUGIN_ID,
  name: 'Skills routed',
  description: 'Shows the skill names the router suggested on the last turn.',

  register(ctx) {
    /**
     * One pane mount. Reads through the plugin's own route and paints the names: the heading always
     * carries the count, the list opens on a click and starts collapsed.
     */
    function SkillsRoutedPane() {
      const expanded = useValue($expanded)
      const { data, error, isPending } = useQuery({
        queryFn: () => ctx.rest(ROUTE),
        queryKey: SUGGESTIONS_QUERY_KEY,
        retry: false
      })
      const missing = isMissingRoute(error)
      const failed = Boolean(error) && !missing
      const pending = Boolean(isPending) && !data && !failed
      const model = summary(missing || failed ? null : data)
      const heading = missing || failed ? PANE_TITLE : model.heading
      const names = missing || failed ? [] : model.names
      const caption = missing || failed || pending ? '' : model.caption
      const note = pending ? '' : missing ? ENABLE_LINE : failed ? READ_ERROR_LINE : model.note

      return jsxs('div', {
        children: [
          jsxs('button', {
            children: [
              jsx(DisclosureCaret, { open: expanded }),
              jsx('span', { children: heading, className: 'font-medium' })
            ],
            className: 'flex w-full items-center gap-1.5 text-left',
            onClick: () => $expanded.set(!$expanded.get()),
            type: 'button'
          }),
          caption
            ? jsx('p', { children: caption, className: 'text-xs text-muted-foreground' })
            : null,
          names.length && expanded
            ? jsx('ul', {
                children: names.map((name, index) =>
                  jsx('li', { children: name, className: 'truncate font-mono text-xs', key: `${index}:${name}` })
                ),
                className: 'flex flex-col gap-1 overflow-y-auto'
              })
            : null,
          names.length || !note
            ? null
            : jsx('p', { children: note, className: 'text-xs text-muted-foreground' })
        ],
        className: 'flex h-full flex-col gap-1 p-3 text-sm'
      })
    }

    ctx.register({
      area: PANES_AREA,
      data: { collapsible: true, placement: 'right', width: PANE_WIDTH },
      id: PANE_ID,
      render: () => jsx(SkillsRoutedPane, {}),
      title: PANE_TITLE
    })

    // The rewrite broadcasts this after a record write; invalidating the query refetches a mounted
    // pane, so a routed turn lands without waiting for the next mount.
    ctx.onEvent(UPDATED_EVENT, () => {
      void queryClient.invalidateQueries({ queryKey: SUGGESTIONS_QUERY_KEY })
    })
  }
}
