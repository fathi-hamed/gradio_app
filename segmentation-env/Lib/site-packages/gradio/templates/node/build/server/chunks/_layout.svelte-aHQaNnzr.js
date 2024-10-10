import { c as create_ssr_component } from './ssr-RaXq3SJh.js';
import { s as svelte, S as SvelteComponentDev } from './ssr-Cql56Qn_.js';
import './Component-Dv7eSVA_.js';

const is_browser = typeof window !== "undefined";
if (is_browser) {
  const o = {
    SvelteComponent: SvelteComponentDev
  };
  for (const key in svelte) {
    if (key === "SvelteComponent")
      continue;
    if (key === "SvelteComponentDev") {
      o[key] = o["SvelteComponent"];
    } else {
      o[key] = svelte[key];
    }
  }
  window.__gradio__svelte__internal = o;
  window.__gradio__svelte__internal["globals"] = {};
  window.globals = window;
}
const Layout = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `${slots.default ? slots.default({}) : ``}`;
});

export { Layout as default };
//# sourceMappingURL=_layout.svelte-aHQaNnzr.js.map
