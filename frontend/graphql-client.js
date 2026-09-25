/**
 * Generic GraphQL client scaffold.
 *
 * No real external GraphQL API exists for this project yet - almost every
 * Indian government data source (LGD, data.gov.in, Census) is REST/CSV, not
 * GraphQL (see README S12, "Integration path"). This file is deliberately
 * just plumbing: a plain fetch()-based client with no endpoint hardcoded,
 * ready to point at a real government or partner GraphQL API once one
 * exists, the same way fact_checker.py's registry lookup is built to have
 * its transport swapped for a live API without changing the calling code.
 *
 * No bundler, no build step, no new npm dependency - loaded the same way
 * supabase-config.js is, with a plain <script> tag.
 */

const GRAPHQL_CONFIG = {
  endpoint: '',   // e.g. 'https://api.example.gov.in/graphql' - unset until a real API exists
  headers: {},    // e.g. { 'X-Api-Key': '...' }
};

/**
 * Run a GraphQL query/mutation against GRAPHQL_CONFIG.endpoint (or an
 * override passed in `options`). Throws a clear, honest error instead of
 * silently sending a request to an unset or placeholder URL - this project
 * never turns "not configured" into "silently did nothing" (see README S3,
 * "Honest degraded mode").
 *
 * @param {string} query - GraphQL query or mutation document.
 * @param {object} [variables] - GraphQL variables.
 * @param {{endpoint?: string, headers?: object}} [options] - per-call overrides.
 * @returns {Promise<object>} the `data` field of the GraphQL response.
 */
async function queryGraphQL(query, variables = {}, options = {}) {
  const endpoint = options.endpoint || GRAPHQL_CONFIG.endpoint;
  if (!endpoint) {
    throw new Error(
      'GraphQL endpoint not configured. Set GRAPHQL_CONFIG.endpoint in '
      + 'graphql-client.js once a real external GraphQL API is available.'
    );
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...GRAPHQL_CONFIG.headers,
      ...options.headers,
    },
    body: JSON.stringify({ query, variables }),
  });

  if (!response.ok) {
    throw new Error(`GraphQL request failed: HTTP ${response.status} ${response.statusText}`);
  }

  const payload = await response.json();
  if (payload.errors && payload.errors.length) {
    throw new Error('GraphQL error: ' + payload.errors.map(e => e.message).join('; '));
  }
  return payload.data;
}

window.GRAPHQL_CONFIG = GRAPHQL_CONFIG;
window.queryGraphQL = queryGraphQL;
