'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var path = require('path');
var purgecss = require('purgecss');

function _interopNamespaceDefault(e) {
    var n = Object.create(null);
    if (e) {
        Object.keys(e).forEach(function (k) {
            if (k !== 'default') {
                var d = Object.getOwnPropertyDescriptor(e, k);
                Object.defineProperty(n, k, d.get ? d : {
                    enumerable: true,
                    get: function () { return e[k]; }
                });
            }
        });
    }
    n.default = e;
    return Object.freeze(n);
}

var path__namespace = /*#__PURE__*/_interopNamespaceDefault(path);

/**
 * PostCSS Plugin for PurgeCSS
 *
 * Most bundlers and frameworks to build websites are using PostCSS.
 * The easiest way to configure PurgeCSS is with its PostCSS plugin.
 *
 * @packageDocumentation
 */
const PLUGIN_NAME = "postcss-purgecss";
/**
 * Execute PurgeCSS process on the postCSS root node
 *
 * @param opts - PurgeCSS options
 * @param root - root node of postCSS
 * @param helpers - postCSS helpers
 */
async function purgeCSS(opts, root, { result }) {
    const purgeCSS = new purgecss.PurgeCSS();
    let configFileOptions;
    try {
        const t = path__namespace.resolve(process.cwd(), "purgecss.config.js");
        configFileOptions = await import(t);
    }
    catch {
        // no config file present
    }
    const options = {
        ...purgecss.defaultOptions,
        ...configFileOptions,
        ...opts,
        safelist: purgecss.standardizeSafelist((opts === null || opts === void 0 ? void 0 : opts.safelist) || (configFileOptions === null || configFileOptions === void 0 ? void 0 : configFileOptions.safelist)),
    };
    if (opts && typeof opts.contentFunction === "function") {
        options.content = opts.contentFunction((root.source && root.source.input.file) || "");
    }
    purgeCSS.options = options;
    if (options.variables) {
        purgeCSS.variablesStructure.safelist = options.safelist.variables || [];
    }
    const { content, extractors } = options;
    const fileFormatContents = content.filter((o) => typeof o === "string");
    const rawFormatContents = content.filter((o) => typeof o === "object");
    const cssFileSelectors = await purgeCSS.extractSelectorsFromFiles(fileFormatContents, extractors);
    const cssRawSelectors = await purgeCSS.extractSelectorsFromString(rawFormatContents, extractors);
    const selectors = purgecss.mergeExtractorSelectors(cssFileSelectors, cssRawSelectors);
    //purge unused selectors
    purgeCSS.walkThroughCSS(root, selectors);
    if (purgeCSS.options.fontFace)
        purgeCSS.removeUnusedFontFaces();
    if (purgeCSS.options.keyframes)
        purgeCSS.removeUnusedKeyframes();
    if (purgeCSS.options.variables)
        purgeCSS.removeUnusedCSSVariables();
    if (purgeCSS.options.rejected && purgeCSS.selectorsRemoved.size > 0) {
        result.messages.push({
            type: "purgecss",
            plugin: "postcss-purgecss",
            text: `purging ${purgeCSS.selectorsRemoved.size} selectors:
          ${Array.from(purgeCSS.selectorsRemoved)
                .map((selector) => selector.trim())
                .join("\n  ")}`,
        });
        purgeCSS.selectorsRemoved.clear();
    }
}
/**
 * PostCSS Plugin for PurgeCSS
 *
 * @param opts - PurgeCSS Options
 * @returns the postCSS plugin
 *
 * @public
 */
const purgeCSSPlugin = function (opts) {
    if (typeof opts === "undefined")
        throw new Error("PurgeCSS plugin does not have the correct options");
    return {
        postcssPlugin: PLUGIN_NAME,
        OnceExit(root, helpers) {
            return purgeCSS(opts, root, helpers);
        },
    };
};
purgeCSSPlugin.postcss = true;

exports.default = purgeCSSPlugin;
exports.purgeCSSPlugin = purgeCSSPlugin;
