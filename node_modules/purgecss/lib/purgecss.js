'use strict';

var fs = require('fs');
var glob = require('glob');
var path = require('path');
var postcss = require('postcss');
var selectorParser = require('postcss-selector-parser');
var util = require('util');

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

var fs__namespace = /*#__PURE__*/_interopNamespaceDefault(fs);
var glob__namespace = /*#__PURE__*/_interopNamespaceDefault(glob);
var path__namespace = /*#__PURE__*/_interopNamespaceDefault(path);
var postcss__namespace = /*#__PURE__*/_interopNamespaceDefault(postcss);

const IGNORE_ANNOTATION_CURRENT = "purgecss ignore current";
const IGNORE_ANNOTATION_NEXT = "purgecss ignore";
const IGNORE_ANNOTATION_START = "purgecss start ignore";
const IGNORE_ANNOTATION_END = "purgecss end ignore";
const CONFIG_FILENAME = "purgecss.config.js";
// Error Message
const ERROR_CONFIG_FILE_LOADING = "Error loading the config file";

function mergeSets(into, from) {
    if (from) {
        from.forEach(into.add, into);
    }
}
/**
 * @public
 */
class ExtractorResultSets {
    constructor(er) {
        this.undetermined = new Set();
        this.attrNames = new Set();
        this.attrValues = new Set();
        this.classes = new Set();
        this.ids = new Set();
        this.tags = new Set();
        this.merge(er);
    }
    merge(that) {
        if (Array.isArray(that)) {
            mergeSets(this.undetermined, that);
        }
        else if (that instanceof ExtractorResultSets) {
            mergeSets(this.undetermined, that.undetermined);
            mergeSets(this.attrNames, that.attrNames);
            mergeSets(this.attrValues, that.attrValues);
            mergeSets(this.classes, that.classes);
            mergeSets(this.ids, that.ids);
            mergeSets(this.tags, that.tags);
        }
        else {
            // ExtractorResultDetailed:
            mergeSets(this.undetermined, that.undetermined);
            if (that.attributes) {
                mergeSets(this.attrNames, that.attributes.names);
                mergeSets(this.attrValues, that.attributes.values);
            }
            mergeSets(this.classes, that.classes);
            mergeSets(this.ids, that.ids);
            mergeSets(this.tags, that.tags);
        }
        return this;
    }
    hasAttrName(name) {
        return this.attrNames.has(name) || this.undetermined.has(name);
    }
    someAttrValue(predicate) {
        for (const val of this.attrValues) {
            if (predicate(val))
                return true;
        }
        for (const val of this.undetermined) {
            if (predicate(val))
                return true;
        }
        return false;
    }
    hasAttrPrefix(prefix) {
        return this.someAttrValue((value) => value.startsWith(prefix));
    }
    hasAttrSuffix(suffix) {
        return this.someAttrValue((value) => value.endsWith(suffix));
    }
    hasAttrSubstr(substr) {
        const wordSubstr = substr.trim().split(" ");
        return wordSubstr.every((word) => this.someAttrValue((value) => value.includes(word)));
    }
    hasAttrValue(value) {
        return this.attrValues.has(value) || this.undetermined.has(value);
    }
    hasClass(name) {
        return this.classes.has(name) || this.undetermined.has(name);
    }
    hasId(id) {
        return this.ids.has(id) || this.undetermined.has(id);
    }
    hasTag(tag) {
        return this.tags.has(tag) || this.undetermined.has(tag);
    }
}

const CSS_SAFELIST = ["*", ":root", ":after", ":before"];

/**
 * @public
 */
const defaultOptions = {
    css: [],
    content: [],
    defaultExtractor: (content) => content.match(/[A-Za-z0-9_-]+/g) || [],
    extractors: [],
    fontFace: false,
    keyframes: false,
    rejected: false,
    rejectedCss: false,
    sourceMap: false,
    stdin: false,
    stdout: false,
    variables: false,
    safelist: {
        standard: [],
        deep: [],
        greedy: [],
        variables: [],
        keyframes: [],
    },
    blocklist: [],
    skippedContentGlobs: [],
    dynamicAttributes: [],
};

/**
 * @public
 */
class VariableNode {
    constructor(declaration) {
        this.nodes = [];
        this.isUsed = false;
        this.value = declaration;
    }
}
/**
 * @public
 */
class VariablesStructure {
    constructor() {
        this.nodes = new Map();
        this.usedVariables = new Set();
        this.safelist = [];
    }
    addVariable(declaration) {
        const { prop } = declaration;
        if (!this.nodes.has(prop)) {
            const node = new VariableNode(declaration);
            this.nodes.set(prop, [node]);
        }
        else {
            const node = new VariableNode(declaration);
            const variableNodes = this.nodes.get(prop) || [];
            this.nodes.set(prop, [...variableNodes, node]);
        }
    }
    addVariableUsage(declaration, matchedVariables) {
        const { prop } = declaration;
        const nodes = this.nodes.get(prop);
        for (const variableMatch of matchedVariables) {
            // capturing group containing the variable is in index 1
            const variableName = variableMatch[1];
            if (this.nodes.has(variableName)) {
                const usedVariableNodes = this.nodes.get(variableName);
                nodes === null || nodes === void 0 ? void 0 : nodes.forEach((node) => {
                    usedVariableNodes === null || usedVariableNodes === void 0 ? void 0 : usedVariableNodes.forEach((usedVariableNode) => node.nodes.push(usedVariableNode));
                });
            }
        }
    }
    addVariableUsageInProperties(matchedVariables) {
        for (const variableMatch of matchedVariables) {
            // capturing group containing the variable is in index 1
            const variableName = variableMatch[1];
            this.usedVariables.add(variableName);
        }
    }
    setAsUsed(variableName) {
        const nodes = this.nodes.get(variableName);
        if (nodes) {
            const queue = [...nodes];
            while (queue.length !== 0) {
                const currentNode = queue.pop();
                if (currentNode && !currentNode.isUsed) {
                    currentNode.isUsed = true;
                    queue.push(...currentNode.nodes);
                }
            }
        }
    }
    removeUnused() {
        // check unordered usage
        for (const used of this.usedVariables) {
            const usedNodes = this.nodes.get(used);
            if (usedNodes) {
                for (const usedNode of usedNodes) {
                    const usedVariablesMatchesInDeclaration = usedNode.value.value.matchAll(/var\((.+?)[,)]/g);
                    for (const usage of usedVariablesMatchesInDeclaration) {
                        if (!this.usedVariables.has(usage[1])) {
                            this.usedVariables.add(usage[1]);
                        }
                    }
                }
            }
        }
        for (const used of this.usedVariables) {
            this.setAsUsed(used);
        }
        for (const [name, declarations] of this.nodes) {
            for (const declaration of declarations) {
                if (!declaration.isUsed && !this.isVariablesSafelisted(name)) {
                    declaration.value.remove();
                }
            }
        }
    }
    isVariablesSafelisted(variable) {
        return this.safelist.some((safelistItem) => {
            return typeof safelistItem === "string"
                ? safelistItem === variable
                : safelistItem.test(variable);
        });
    }
}

/**
 * Core package of PurgeCSS
 *
 * Contains the core methods to analyze the files, remove unused CSS.
 *
 * @packageDocumentation
 */
const asyncFs = {
    access: util.promisify(fs__namespace.access),
    readFile: util.promisify(fs__namespace.readFile),
};
/**
 * Format the user defined safelist into a standardized safelist object
 *
 * @param userDefinedSafelist - the user defined safelist
 * @returns the formatted safelist object that can be used in the PurgeCSS options
 *
 * @public
 */
function standardizeSafelist(userDefinedSafelist = []) {
    if (Array.isArray(userDefinedSafelist)) {
        return {
            ...defaultOptions.safelist,
            standard: userDefinedSafelist,
        };
    }
    return {
        ...defaultOptions.safelist,
        ...userDefinedSafelist,
    };
}
/**
 * Load the configuration file from the path
 *
 * @param configFile - Path of the config file
 * @returns The options from the configuration file
 *
 * @throws Error
 * This exception is thrown if the configuration file was not imported
 *
 * @public
 */
async function setOptions(configFile = CONFIG_FILENAME) {
    let options;
    try {
        const t = path__namespace.resolve(process.cwd(), configFile);
        options = await import(t);
    }
    catch (err) {
        if (err instanceof Error) {
            throw new Error(`${ERROR_CONFIG_FILE_LOADING} ${err.message}`);
        }
        throw new Error();
    }
    return {
        ...defaultOptions,
        ...options,
        safelist: standardizeSafelist(options.safelist),
    };
}
/**
 * Use the extract function to get the list of selectors
 *
 * @param content - content (e.g. html file)
 * @param extractor - PurgeCSS extractor used to extract the selectors
 * @returns the sets containing the result of the extractor function
 */
async function extractSelectors(content, extractor) {
    return new ExtractorResultSets(await extractor(content));
}
/**
 * Check if the node is a css comment indication to ignore the selector rule
 *
 * @param node - node of postcss AST
 * @param type - type of css comment
 * @returns true if the node is a PurgeCSS ignore comment
 */
function isIgnoreAnnotation(node, type) {
    switch (type) {
        case "next":
            return node.text.includes(IGNORE_ANNOTATION_NEXT);
        case "start":
            return node.text.includes(IGNORE_ANNOTATION_START);
        case "end":
            return node.text.includes(IGNORE_ANNOTATION_END);
    }
}
/**
 * Check if the node correspond to an empty css rule
 *
 * @param node - node of postcss AST
 * @returns true if the rule is empty
 */
function isRuleEmpty(node) {
    if ((isPostCSSRule(node) && !node.selector) ||
        ((node === null || node === void 0 ? void 0 : node.nodes) && !node.nodes.length) ||
        (isPostCSSAtRule(node) &&
            ((!node.nodes && !node.params) ||
                (!node.params && node.nodes && !node.nodes.length)))) {
        return true;
    }
    return false;
}
/**
 * Check if the node has a css comment indicating to ignore the current selector rule
 *
 * @param rule - rule of postcss AST
 */
function hasIgnoreAnnotation(rule) {
    let found = false;
    rule.walkComments((node) => {
        if (node &&
            node.type === "comment" &&
            node.text.includes(IGNORE_ANNOTATION_CURRENT)) {
            found = true;
            node.remove();
        }
    });
    return found;
}
/**
 * Merge two extractor selectors
 *
 * @param extractorSelectorsA - extractor selectors A
 * @param extractorSelectorsB - extractor selectors B
 * @returns  the merged extractor result sets
 *
 * @public
 */
function mergeExtractorSelectors(...extractors) {
    const result = new ExtractorResultSets([]);
    extractors.forEach(result.merge, result);
    return result;
}
/**
 * Strips quotes of a string
 *
 * @param str - string to be stripped
 */
function stripQuotes(str) {
    return str.replace(/(^["'])|(["']$)/g, "");
}
/**
 * Returns true if the attribute is found in the extractor selectors
 *
 * @param attributeNode - node of type `attribute`
 * @param selectors - extractor selectors
 */
function isAttributeFound(attributeNode, selectors) {
    if (!selectors.hasAttrName(attributeNode.attribute)) {
        return false;
    }
    if (typeof attributeNode.value === "undefined") {
        return true;
    }
    switch (attributeNode.operator) {
        case "$=":
            return selectors.hasAttrSuffix(attributeNode.value);
        case "~=":
        case "*=":
            return selectors.hasAttrSubstr(attributeNode.value);
        case "=":
            return selectors.hasAttrValue(attributeNode.value);
        case "|=":
        case "^=":
            return selectors.hasAttrPrefix(attributeNode.value);
        default:
            return true;
    }
}
/**
 * Returns true if the class is found in the extractor selectors
 *
 * @param classNode - node of type `class`
 * @param selectors - extractor selectors
 */
function isClassFound(classNode, selectors) {
    return selectors.hasClass(classNode.value);
}
/**
 * Returns true if the identifier is found in the extractor selectors
 *
 * @param identifierNode - node of type `identifier`
 * @param selectors - extractor selectors
 */
function isIdentifierFound(identifierNode, selectors) {
    return selectors.hasId(identifierNode.value);
}
/**
 * Returns true if the tag is found in the extractor selectors
 *
 * @param tagNode - node of type `tag`
 * @param selectors - extractor selectors
 */
function isTagFound(tagNode, selectors) {
    return selectors.hasTag(tagNode.value);
}
/**
 * Returns true if the selector is inside a pseudo class
 * (e.g. :nth-child, :nth-of-type, :only-child, :not)
 *
 * @param selector - selector
 */
function isInPseudoClass(selector) {
    return ((selector.parent &&
        selector.parent.type === "pseudo" &&
        selector.parent.value.startsWith(":")) ||
        false);
}
/**
 * Returns true if the selector is inside the pseudo classes :where() or :is()
 * @param selector - selector
 */
function isInPseudoClassWhereOrIs(selector) {
    return ((selector.parent &&
        selector.parent.type === "pseudo" &&
        (selector.parent.value === ":where" ||
            selector.parent.value === ":is")) ||
        false);
}
/**
 * Returns true if the selector is a pseudo class at the root level
 * Pseudo classes checked: :where, :is, :has, :not
 * @param selector - selector
 */
function isPseudoClassAtRootLevel(selector) {
    var _a;
    let result = false;
    if (selector.type === "selector" &&
        ((_a = selector.parent) === null || _a === void 0 ? void 0 : _a.type) === "root" &&
        selector.nodes.length === 1) {
        selector.walk((node) => {
            if (node.type === "pseudo" &&
                (node.value === ":where" ||
                    node.value === ":is" ||
                    node.value === ":has" ||
                    node.value === ":not")) {
                result = true;
            }
        });
    }
    return result;
}
function isPostCSSAtRule(node) {
    return (node === null || node === void 0 ? void 0 : node.type) === "atrule";
}
function isPostCSSRule(node) {
    return (node === null || node === void 0 ? void 0 : node.type) === "rule";
}
function isPostCSSComment(node) {
    return (node === null || node === void 0 ? void 0 : node.type) === "comment";
}
/**
 * Class used to instantiate PurgeCSS and can then be used
 * to purge CSS files.
 *
 * @example
 * ```ts
 * await new PurgeCSS().purge({
 *    content: ['index.html'],
 *    css: ['css/app.css']
 * })
 * ```
 *
 * @public
 */
class PurgeCSS {
    constructor() {
        this.ignore = false;
        this.atRules = {
            fontFace: [],
            keyframes: [],
        };
        this.usedAnimations = new Set();
        this.usedFontFaces = new Set();
        this.selectorsRemoved = new Set();
        this.removedNodes = [];
        this.variablesStructure = new VariablesStructure();
        this.options = defaultOptions;
    }
    collectDeclarationsData(declaration) {
        const { prop, value } = declaration;
        // collect css properties data
        if (this.options.variables) {
            const usedVariablesMatchesInDeclaration = value.matchAll(/var\((.+?)[,)]/g);
            if (prop.startsWith("--")) {
                this.variablesStructure.addVariable(declaration);
                this.variablesStructure.addVariableUsage(declaration, usedVariablesMatchesInDeclaration);
            }
            else {
                this.variablesStructure.addVariableUsageInProperties(usedVariablesMatchesInDeclaration);
            }
        }
        // collect keyframes data
        if (this.options.keyframes) {
            if (prop === "animation" || prop === "animation-name") {
                for (const word of value.split(/[\s,]+/)) {
                    this.usedAnimations.add(word);
                }
                return;
            }
        }
        // collect font faces data
        if (this.options.fontFace) {
            if (prop === "font-family") {
                for (const fontName of value.split(",")) {
                    const cleanedFontFace = stripQuotes(fontName.trim());
                    this.usedFontFaces.add(cleanedFontFace);
                }
            }
            return;
        }
    }
    /**
     * Get the extractor corresponding to the extension file
     * @param filename - Name of the file
     * @param extractors - Array of extractors definition
     */
    getFileExtractor(filename, extractors) {
        const extractorObj = extractors.find((extractor) => extractor.extensions.find((ext) => filename.endsWith(ext)));
        return typeof extractorObj === "undefined"
            ? this.options.defaultExtractor
            : extractorObj.extractor;
    }
    /**
     * Extract the selectors present in the files using a PurgeCSS extractor
     *
     * @param files - Array of files path or glob pattern
     * @param extractors - Array of extractors
     */
    async extractSelectorsFromFiles(files, extractors) {
        const selectors = new ExtractorResultSets([]);
        const filesNames = [];
        for (const globFile of files) {
            try {
                await asyncFs.access(globFile, fs__namespace.constants.F_OK);
                filesNames.push(globFile);
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
            }
            catch (err) {
                filesNames.push(...glob__namespace.sync(globFile, {
                    nodir: true,
                    ignore: this.options.skippedContentGlobs.map((glob) => glob.replace(/^\.\//, "")),
                }));
            }
        }
        if (files.length > 0 && filesNames.length === 0) {
            console.warn("No files found from the passed PurgeCSS option 'content'.");
        }
        for (const file of filesNames) {
            const content = await asyncFs.readFile(file, "utf-8");
            const extractor = this.getFileExtractor(file, extractors);
            const extractedSelectors = await extractSelectors(content, extractor);
            selectors.merge(extractedSelectors);
        }
        return selectors;
    }
    /**
     * Extract the selectors present in the passed string using a PurgeCSS extractor
     *
     * @param content - Array of content
     * @param extractors - Array of extractors
     */
    async extractSelectorsFromString(content, extractors) {
        const selectors = new ExtractorResultSets([]);
        for (const { raw, extension } of content) {
            const extractor = this.getFileExtractor(`.${extension}`, extractors);
            const extractedSelectors = await extractSelectors(raw, extractor);
            selectors.merge(extractedSelectors);
        }
        return selectors;
    }
    /**
     * Evaluate at-rule and register it for future reference
     * @param node - node of postcss AST
     */
    evaluateAtRule(node) {
        // keyframes
        if (this.options.keyframes && node.name.endsWith("keyframes")) {
            this.atRules.keyframes.push(node);
            return;
        }
        // font-face
        if (this.options.fontFace && node.name === "font-face" && node.nodes) {
            for (const childNode of node.nodes) {
                if (childNode.type === "decl" && childNode.prop === "font-family") {
                    this.atRules.fontFace.push({
                        name: stripQuotes(childNode.value),
                        node,
                    });
                }
            }
        }
    }
    /**
     * Evaluate css selector and decide if it should be removed or not
     *
     * @param node - node of postcss AST
     * @param selectors - selectors used in content files
     */
    evaluateRule(node, selectors) {
        // exit if is in ignoring state activated by an ignore comment
        if (this.ignore) {
            return;
        }
        // exit if the previous annotation is a ignore next line comment
        const annotation = node.prev();
        if (isPostCSSComment(annotation) &&
            isIgnoreAnnotation(annotation, "next")) {
            annotation.remove();
            return;
        }
        // exit if it is inside a keyframes
        if (node.parent &&
            isPostCSSAtRule(node.parent) &&
            node.parent.name.endsWith("keyframes")) {
            return;
        }
        // exit if it is not a rule
        if (!isPostCSSRule(node)) {
            return;
        }
        // exit if it has an ignore rule comment inside
        if (hasIgnoreAnnotation(node)) {
            return;
        }
        const selectorsRemovedFromRule = [];
        // selector transformer, walk over the list of the parsed selectors twice.
        // First pass will remove the unused selectors. It goes through
        // pseudo-classes like :where() and :is() and remove the unused
        // selectors inside of them, but will not remove the pseudo-classes
        // themselves. Second pass will remove selectors containing empty
        // :where and :is.
        node.selector = selectorParser((selectorsParsed) => {
            selectorsParsed.walk((selector) => {
                if (selector.type !== "selector") {
                    return;
                }
                const keepSelector = this.shouldKeepSelector(selector, selectors);
                if (!keepSelector) {
                    if (this.options.rejected) {
                        this.selectorsRemoved.add(selector.toString());
                    }
                    if (this.options.rejectedCss) {
                        selectorsRemovedFromRule.push(selector.toString());
                    }
                    selector.remove();
                }
            });
            // removes selectors containing empty :where and :is
            selectorsParsed.walk((selector) => {
                if (selector.type !== "selector") {
                    return;
                }
                if (selector.toString() && /(:where)|(:is)/.test(selector.toString())) {
                    selector.walk((node) => {
                        if (node.type !== "pseudo")
                            return;
                        if (node.value !== ":where" && node.value !== ":is")
                            return;
                        if (node.nodes.length === 0) {
                            selector.remove();
                        }
                    });
                }
            });
        }).processSync(node.selector);
        // declarations
        if (node.selector && typeof node.nodes !== "undefined") {
            for (const childNode of node.nodes) {
                if (childNode.type !== "decl")
                    continue;
                this.collectDeclarationsData(childNode);
            }
        }
        // remove empty rules
        const parent = node.parent;
        if (!node.selector) {
            node.remove();
        }
        if (isRuleEmpty(parent))
            parent === null || parent === void 0 ? void 0 : parent.remove();
        // rebuild the rule with the removed selectors and optionally its parent
        if (this.options.rejectedCss) {
            if (selectorsRemovedFromRule.length > 0) {
                const clone = node.clone();
                const parentClone = parent === null || parent === void 0 ? void 0 : parent.clone().removeAll().append(clone);
                clone.selectors = selectorsRemovedFromRule;
                const nodeToPreserve = parentClone ? parentClone : clone;
                this.removedNodes.push(nodeToPreserve);
            }
        }
    }
    /**
     * Get the purged version of the css based on the files
     *
     * @param cssOptions - css options, files or raw strings
     * @param selectors - set of extracted css selectors
     */
    async getPurgedCSS(cssOptions, selectors) {
        var _a;
        const sources = [];
        // resolve any globs
        const processedOptions = [];
        for (const option of cssOptions) {
            if (typeof option === "string") {
                processedOptions.push(...glob__namespace.sync(option, {
                    nodir: true,
                    ignore: this.options.skippedContentGlobs,
                }));
            }
            else {
                processedOptions.push(option);
            }
        }
        for (const option of processedOptions) {
            const cssContent = typeof option === "string"
                ? this.options.stdin
                    ? option
                    : await asyncFs.readFile(option, "utf-8")
                : option.raw;
            const isFromFile = typeof option === "string" && !this.options.stdin;
            const root = postcss__namespace.parse(cssContent, {
                from: isFromFile ? option : undefined,
            });
            // purge unused selectors
            this.walkThroughCSS(root, selectors);
            if (this.options.fontFace)
                this.removeUnusedFontFaces();
            if (this.options.keyframes)
                this.removeUnusedKeyframes();
            if (this.options.variables)
                this.removeUnusedCSSVariables();
            const postCSSResult = root.toResult({
                map: this.options.sourceMap,
                to: typeof this.options.sourceMap === "object"
                    ? this.options.sourceMap.to
                    : undefined,
            });
            const result = {
                css: postCSSResult.toString(),
                file: typeof option === "string" ? option : option.name,
            };
            if (this.options.sourceMap) {
                result.sourceMap = (_a = postCSSResult.map) === null || _a === void 0 ? void 0 : _a.toString();
            }
            if (this.options.rejected) {
                result.rejected = Array.from(this.selectorsRemoved);
                this.selectorsRemoved.clear();
            }
            if (this.options.rejectedCss) {
                result.rejectedCss = postcss__namespace
                    .root({ nodes: this.removedNodes })
                    .toString();
            }
            sources.push(result);
        }
        return sources;
    }
    /**
     * Check if the keyframe is safelisted with the option safelist keyframes
     *
     * @param keyframesName - name of the keyframe animation
     */
    isKeyframesSafelisted(keyframesName) {
        return this.options.safelist.keyframes.some((safelistItem) => {
            return typeof safelistItem === "string"
                ? safelistItem === keyframesName
                : safelistItem.test(keyframesName);
        });
    }
    /**
     * Check if the selector is blocklisted with the option blocklist
     *
     * @param selector - css selector
     */
    isSelectorBlocklisted(selector) {
        return this.options.blocklist.some((blocklistItem) => {
            return typeof blocklistItem === "string"
                ? blocklistItem === selector
                : blocklistItem.test(selector);
        });
    }
    /**
     * Check if the selector is safelisted with the option safelist standard
     *
     * @param selector - css selector
     */
    isSelectorSafelisted(selector) {
        const isSafelisted = this.options.safelist.standard.some((safelistItem) => {
            return typeof safelistItem === "string"
                ? safelistItem === selector
                : safelistItem.test(selector);
        });
        const isPseudoElement = /^::.*/.test(selector);
        return CSS_SAFELIST.includes(selector) || isPseudoElement || isSafelisted;
    }
    /**
     * Check if the selector is safelisted with the option safelist deep
     *
     * @param selector - selector
     */
    isSelectorSafelistedDeep(selector) {
        return this.options.safelist.deep.some((safelistItem) => safelistItem.test(selector));
    }
    /**
     * Check if the selector is safelisted with the option safelist greedy
     *
     * @param selector - selector
     */
    isSelectorSafelistedGreedy(selector) {
        return this.options.safelist.greedy.some((safelistItem) => safelistItem.test(selector));
    }
    /**
     * Remove unused CSS
     *
     * @param userOptions - PurgeCSS options or path to the configuration file
     * @returns an array of object containing the filename and the associated CSS
     *
     * @example Using a configuration file named purgecss.config.js
     * ```ts
     * const purgeCSSResults = await new PurgeCSS().purge()
     * ```
     *
     * @example Using a custom path to the configuration file
     * ```ts
     * const purgeCSSResults = await new PurgeCSS().purge('./purgecss.config.js')
     * ```
     *
     * @example Using the PurgeCSS options
     * ```ts
     * const purgeCSSResults = await new PurgeCSS().purge({
     *    content: ['index.html', '**\/*.js', '**\/*.html', '**\/*.vue'],
     *    css: ['css/app.css']
     * })
     * ```
     */
    async purge(userOptions) {
        this.options =
            typeof userOptions !== "object"
                ? await setOptions(userOptions)
                : {
                    ...defaultOptions,
                    ...userOptions,
                    safelist: standardizeSafelist(userOptions.safelist),
                };
        const { content, css, extractors, safelist } = this.options;
        if (this.options.variables) {
            this.variablesStructure.safelist = safelist.variables || [];
        }
        const fileFormatContents = content.filter((o) => typeof o === "string");
        const rawFormatContents = content.filter((o) => typeof o === "object");
        const cssFileSelectors = await this.extractSelectorsFromFiles(fileFormatContents, extractors);
        const cssRawSelectors = await this.extractSelectorsFromString(rawFormatContents, extractors);
        return this.getPurgedCSS(css, mergeExtractorSelectors(cssFileSelectors, cssRawSelectors));
    }
    /**
     * Remove unused CSS variables
     */
    removeUnusedCSSVariables() {
        this.variablesStructure.removeUnused();
    }
    /**
     * Remove unused font-faces
     */
    removeUnusedFontFaces() {
        for (const { name, node } of this.atRules.fontFace) {
            if (!this.usedFontFaces.has(name)) {
                node.remove();
            }
        }
    }
    /**
     * Remove unused keyframes
     */
    removeUnusedKeyframes() {
        for (const node of this.atRules.keyframes) {
            if (!this.usedAnimations.has(node.params) &&
                !this.isKeyframesSafelisted(node.params)) {
                node.remove();
            }
        }
    }
    /**
     * Transform a selector node into a string
     */
    getSelectorValue(selector) {
        return ((selector.type === "attribute" && selector.attribute) || selector.value);
    }
    /**
     * Determine if the selector should be kept, based on the selectors found in the files
     *
     * @param selector - set of css selectors found in the content files or string
     * @param selectorsFromExtractor - selectors in the css rule
     *
     * @returns true if the selector should be kept in the processed CSS
     */
    shouldKeepSelector(selector, selectorsFromExtractor) {
        // selectors in pseudo classes are ignored except :where() and :is(). For those pseudo-classes, we are treating the selectors inside the same way as they would be outside.
        if (isInPseudoClass(selector) && !isInPseudoClassWhereOrIs(selector)) {
            return true;
        }
        if (isPseudoClassAtRootLevel(selector)) {
            return true;
        }
        // if there is any greedy safelist pattern, run all the selector parts through them
        // if there is any match, return true
        if (this.options.safelist.greedy.length > 0) {
            const selectorParts = selector.nodes.map(this.getSelectorValue);
            if (selectorParts.some((selectorPart) => selectorPart && this.isSelectorSafelistedGreedy(selectorPart))) {
                return true;
            }
        }
        let isPresent = false;
        for (const selectorNode of selector.nodes) {
            const selectorValue = this.getSelectorValue(selectorNode);
            // if the selector is safelisted with children
            // returns true to keep all children selectors
            if (selectorValue && this.isSelectorSafelistedDeep(selectorValue)) {
                return true;
            }
            // The selector is found in the internal and user-defined safelist
            if (selectorValue &&
                (CSS_SAFELIST.includes(selectorValue) ||
                    this.isSelectorSafelisted(selectorValue))) {
                isPresent = true;
                continue;
            }
            // The selector is present in the blocklist
            if (selectorValue && this.isSelectorBlocklisted(selectorValue)) {
                return false;
            }
            switch (selectorNode.type) {
                case "attribute":
                    // `value` is a dynamic attribute, highly used in input element
                    // the choice is to always leave `value` as it can change based on the user
                    // idem for `checked`, `selected`, `open`
                    isPresent = [
                        ...this.options.dynamicAttributes,
                        "value",
                        "checked",
                        "selected",
                        "open",
                    ].includes(selectorNode.attribute)
                        ? true
                        : isAttributeFound(selectorNode, selectorsFromExtractor);
                    break;
                case "class":
                    isPresent = isClassFound(selectorNode, selectorsFromExtractor);
                    break;
                case "id":
                    isPresent = isIdentifierFound(selectorNode, selectorsFromExtractor);
                    break;
                case "tag":
                    isPresent = isTagFound(selectorNode, selectorsFromExtractor);
                    break;
                default:
                    continue;
            }
            // selector is not safelisted
            // and it has not been found as an attribute/class/id/tag
            if (!isPresent) {
                return false;
            }
        }
        return isPresent;
    }
    /**
     * Walk through the CSS AST and remove unused CSS
     *
     * @param root - root node of the postcss AST
     * @param selectors - selectors used in content files
     */
    walkThroughCSS(root, selectors) {
        root.walk((node) => {
            if (node.type === "rule") {
                return this.evaluateRule(node, selectors);
            }
            if (node.type === "atrule") {
                return this.evaluateAtRule(node);
            }
            if (node.type === "comment") {
                if (isIgnoreAnnotation(node, "start")) {
                    this.ignore = true;
                    // remove ignore annotation
                    node.remove();
                }
                else if (isIgnoreAnnotation(node, "end")) {
                    this.ignore = false;
                    // remove ignore annotation
                    node.remove();
                }
            }
        });
    }
}

exports.ExtractorResultSets = ExtractorResultSets;
exports.PurgeCSS = PurgeCSS;
exports.VariableNode = VariableNode;
exports.VariablesStructure = VariablesStructure;
exports.defaultOptions = defaultOptions;
exports.mergeExtractorSelectors = mergeExtractorSelectors;
exports.setOptions = setOptions;
exports.standardizeSafelist = standardizeSafelist;
