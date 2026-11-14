/**
 * Skills routed — the Ink TUI half of the display (module M14, contract C9).
 *
 * Hermes gives plugins no display hook in the Ink TUI, so this half is a USER widget the plugin
 * ships: symlink/copy it into `$HERMES_HOME/tui-widgets/skill-router.mjs` (T7 documents the step)
 * and the registry picks it up as `/skills-routed`. User widget modules are imported into the TUI
 * process and run with its privileges, and the host passes only the widget SDK — so the widget
 * reads the display record itself with `node:fs` instead of asking the plugin. It takes the record
 * path as its launch argument and defaults to the same path the plugin writes by default; it never
 * reads plugin config, which is why a disabled display is simply a missing record.
 *
 * It shows the skill names the router suggested on the last turn, names only, with one muted caption
 * naming the turn — or the model search, when the record says a model called the search tool — and
 * its local clock time from the record's own identifiers. Its other muted line is the record's own
 * `note`: contract C5's one line for an incomplete history that is healing, which the plugin wrote
 * and a request never carried. The record's eight fields are projected down to status, origin,
 * names, note, timestamp and turn id, so text from a request can never reach the card. It re-reads
 * the file every 500 ms and lands each update through `sdk.updateWidget`, which no-ops once the
 * widget is closed, and the dock's own launch toggle collapses it on a second `/skills-routed`. A
 * missing, corrupt or `idle` record paints one line.
 */

import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

/** The widget app id: the slash command is `/skills-routed`. */
export const WIDGET_ID = 'skills-routed'

/** Where the plugin writes the record unless `suggestionsPath` is configured otherwise. */
export const DEFAULT_RECORD_PATH = join(homedir(), '.cache', 'hermes-skill-router', 'suggestions.json')

/** Fixed card width in cells; the dock reserves exactly this rail. */
export const WIDTH = 44

/** The card title and the collapsible section's label. */
export const TITLE = 'Skills routed'
export const ACCORDION_TITLE = 'Suggestions'

/** How often the widget re-reads the record file. */
export const POLL_MS = 500

/** The record's current schema; a schema 1 file still reads, as a turn record. */
const SCHEMA = 2
const LEGACY_SCHEMA = 1

/** Who produced the record: the automatic turn pass, or a model call to the search tool. */
const ORIGINS = ['turn', 'search']

/** The marker a search record paints where a routed turn paints its turn id. */
export const MODEL_SEARCH_LABEL = 'model search'

/** The two decisions a record may carry. */
const STATUSES = ['routed', 'inert']

const IDLE_LINE = 'no suggestions recorded yet'
const INERT_LINE = 'router inactive: no suggestions this turn'
const EMPTY_LINE = 'the last routed turn suggested nothing'

/** The idle answer for a record that is absent, unreadable or not a record. */
export function idleRecord() {
  return { at: '', names: [], note: '', origin: '', status: 'idle', turn_id: '' }
}

/**
 * Read one record file into `{status, names, at, turn_id, origin, note}`; never throws.
 *
 * Absent, unreadable, non-JSON, wrong-schema or wrong-shaped files all return :func:`idleRecord`,
 * so the widget has exactly one shape to render. A schema 1 file still reads, with origin `turn`,
 * so an installed older writer does not blank a newer card. Only those six fields leave this
 * function: the names are the record's own suggestion names (a kept row is never in the file), the
 * note is the one line the plugin wrote for C5, and no message, prompt or any other text is read
 * out of it.
 */
export function readRecord(file) {
  let data

  try {
    data = JSON.parse(readFileSync(file, 'utf8'))
  } catch {
    return idleRecord()
  }

  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return idleRecord()
  }

  if (![LEGACY_SCHEMA, SCHEMA].includes(data.schema) || !STATUSES.includes(data.status)) {
    return idleRecord()
  }

  const origin = data.schema === LEGACY_SCHEMA ? 'turn' : data.origin

  if (!ORIGINS.includes(origin)) {
    return idleRecord()
  }

  if (!Array.isArray(data.names) || data.names.some(name => typeof name !== 'string')) {
    return idleRecord()
  }

  return {
    at: typeof data.at === 'string' ? data.at : '',
    names: data.status === 'routed' ? data.names.filter(name => name.trim() !== '') : [],
    note: typeof data.note === 'string' ? data.note.trim() : '',
    origin,
    status: data.status,
    turn_id: typeof data.turn_id === 'string' ? data.turn_id : ''
  }
}

/**
 * The local clock time of an ISO 8601 record timestamp, `HH:MM`, or `""` when it does not parse.
 *
 * The record stores UTC and the card shows the reader's own clock: the caption answers "when did
 * this decision happen for me", not "what did the writer's clock say".
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
 * The one caption line above the list: which decision the names came from and when.
 *
 * Built from the record's own `origin`, `turn_id` and `at` only — never from a message, a prompt or
 * any other field — and empty for `inert` and for an absent record, so a stale turn id can never
 * sit under an inactive or idle state. A `search` record paints the model-search marker instead of
 * a turn id, because a lookup belongs to no turn: `model search · <local HH:MM>`. A `turn` record
 * keeps `turn <id> · <local HH:MM>`, dropping either half when it is missing.
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
 * The labels one render needs, derived from a record: the section title, its count, the names to
 * list one per line, the turn caption, and the single line a state without names paints.
 *
 * `note` is the record's own C5 line when the writer supplied one, and otherwise the fixed line
 * this state has always painted: an inert turn, an idle widget or a routed turn that picked
 * nothing. A routed turn that picked names carries no line unless a note says history is healing,
 * so the card gains one line while the association store is incomplete and stays silent once it
 * is complete.
 */
export function summary(record) {
  const status = record && STATUSES.includes(record.status) ? record.status : 'idle'
  const names = status === 'routed' && Array.isArray(record.names) ? record.names : []
  const caption = captionFor(record)
  const note =
    record && typeof record === 'object' && typeof record.note === 'string' ? record.note.trim() : ''

  if (status === 'inert') {
    return {
      caption,
      count: 0,
      names: [],
      note: note || INERT_LINE,
      status: 'inert',
      title: ACCORDION_TITLE
    }
  }

  if (status !== 'routed') {
    return {
      caption,
      count: 0,
      names: [],
      note: note || IDLE_LINE,
      status: 'idle',
      title: ACCORDION_TITLE
    }
  }

  return {
    caption,
    count: names.length,
    names,
    note: note || (names.length ? '' : EMPTY_LINE),
    status: 'routed',
    title: ACCORDION_TITLE
  }
}

/** Resolve the launch argument: blank means the plugin's default path, `~` means the home dir. */
export function expandPath(argument) {
  const text = String(argument ?? '').trim()

  if (!text) {
    return DEFAULT_RECORD_PATH
  }

  if (text === '~') {
    return homedir()
  }

  return text.startsWith('~/') ? join(homedir(), text.slice(2)) : text
}

/** The default path as the help line spells it (`~` instead of this machine's home dir). */
export function helpPath(path = DEFAULT_RECORD_PATH) {
  const home = homedir()

  return path === home ? '~' : path.startsWith(`${home}/`) ? `~${path.slice(home.length)}` : path
}

export default function register(sdk) {
  let app = null
  let timer = null

  /** Paint the card: a title, the collapsible name list and C5's note line when there is one. */
  function face(state, t) {
    const model = summary(state.record)
    const caption = model.caption ? sdk.h(sdk.Text, { color: t.color.muted }, model.caption) : null
    const list = model.names.length
      ? sdk.h(
          sdk.Accordion,
          { count: model.count, defaultOpen: false, t, title: model.title },
          model.names.map((name, index) => {
            return sdk.h(sdk.Text, { color: t.color.label, key: `${index}:${name}` }, name)
          })
        )
      : null
    const note = model.note ? sdk.h(sdk.Text, { color: t.color.muted }, model.note) : null
    const body = [list, note].filter(Boolean)

    return sdk.h(
      sdk.Dialog,
      { title: TITLE, width: WIDTH },
      caption,
      body.length === 1 ? body[0] : body
    )
  }

  app = sdk.defineWidgetApp({
    help: `skill names the router suggested last turn (arg: record file, default ${helpPath()})`,

    id: WIDGET_ID,

    /**
     * Take the record path, paint what is there now, then re-read it every POLL_MS. One timer per
     * launch: a relaunch replaces the previous one so a collapsed card cannot leave a poller behind.
     */
    init(argument) {
      const path = expandPath(argument)

      if (timer) {
        clearInterval(timer)
      }

      timer = setInterval(() => {
        sdk.updateWidget(app, state => ({ ...state, record: readRecord(path) }))
      }, POLL_MS)
      timer.unref?.()

      return { path, record: readRecord(path) }
    },

    mode: 'ambient',

    // Ambient apps capture no input; the launch toggle is their dismissal.
    reduce(state) {
      return state
    },

    render({ state, t }) {
      return face(state, t)
    },

    usage: `usage: /${WIDGET_ID} [record.json]   (default ${DEFAULT_RECORD_PATH})`,

    width: WIDTH
  })
}
