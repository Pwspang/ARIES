// Idempotent Milvus RBAC for the CodeRAG index:
//   coderag_build: create/drop/insert collections in the default database (offline build only)
//   coderag_serve: read-only (describe/list, load, query, search) for the MCP server
// Env: MILVUS_ADDRESS, MILVUS_ROOT_PASSWORD, CODERAG_BUILD_PASSWORD, CODERAG_SERVE_PASSWORD
import { MilvusClient } from "@zilliz/milvus2-sdk-node";

const address = process.env.MILVUS_ADDRESS;
const rootPassword = process.env.MILVUS_ROOT_PASSWORD;
const need = (k) => { if (!process.env[k]) throw new Error(`${k} is required`); return process.env[k]; };
need("MILVUS_ADDRESS"); need("MILVUS_ROOT_PASSWORD");

// First run: root still has the default password; rotate it.
// The constructor connects eagerly; await connectPromise so a wrong password is a
// catchable rejection rather than an unhandled one.
const connect = async (token) => {
  const c = new MilvusClient({ address, token });
  await c.connectPromise;
  return c;
};
let client;
try {
  client = await connect(`root:${rootPassword}`);
} catch {
  const bootstrap = await connect("root:Milvus");
  const res = await bootstrap.updateUser({ username: "root", oldPassword: "Milvus", newPassword: rootPassword });
  if (res.error_code !== "Success") throw new Error(`rotate root password: ${JSON.stringify(res)}`);
  await bootstrap.closeConnection();
  client = await connect(`root:${rootPassword}`);
}
const ok = (what, res) => {
  const code = res?.error_code ?? res?.status?.error_code;
  if (code && code !== "Success") throw new Error(`${what}: ${JSON.stringify(res)}`);
};

const users = new Set((await client.listUsers()).usernames);
const roles = {
  coderag_build_role: {
    user: "coderag_build",
    password: need("CODERAG_BUILD_PASSWORD"),
    grants: [
      ["DatabaseAdmin", "default", "*"],
      ["CollectionAdmin", "default", "*"],
    ],
  },
  coderag_serve_role: {
    user: "coderag_serve",
    password: need("CODERAG_SERVE_PASSWORD"),
    grants: [
      ["DatabaseReadOnly", "default", "*"],
      ["CollectionReadOnly", "default", "*"],
      ["Load", "default", "*"],
      ["GetLoadState", "default", "*"],
    ],
  },
};
for (const [role, spec] of Object.entries(roles)) {
  if (!users.has(spec.user)) ok(`create ${spec.user}`, await client.createUser({ username: spec.user, password: spec.password }));
  const existing = await client.describeRole({ roleName: role }).catch(() => null);
  if (!existing || existing.status?.error_code !== "Success" || !existing.results?.length) {
    ok(`create ${role}`, await client.createRole({ roleName: role }));
  }
  for (const [privilege, db_name, collection_name] of spec.grants) {
    ok(`grant ${privilege} to ${role}`, await client.grantPrivilegeV2({ role, privilege, db_name, collection_name }));
  }
  ok(`bind ${spec.user}`, await client.addUserToRole({ username: spec.user, roleName: role }));
}
console.log(JSON.stringify({ users: (await client.listUsers()).usernames }));
await client.closeConnection();
