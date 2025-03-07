'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var path = require('path');
var esbuild = require('esbuild');
var loaderUtils = require('loader-utils');
require('webpack');
var getTsconfig = require('get-tsconfig');
var webpackSources = require('webpack-sources');
var ModuleFilenameHelpers = require('webpack/lib/ModuleFilenameHelpers.js');

const tsconfigCache = /* @__PURE__ */ new Map();
const tsExtensionsPattern = /\.(?:[cm]?ts|[tj]sx)$/;
async function ESBuildLoader(source) {
  const done = this.async();
  const options = typeof this.getOptions === "function" ? this.getOptions() : loaderUtils.getOptions(this);
  const {
    implementation,
    tsconfig: tsconfigPath,
    ...esbuildTransformOptions
  } = options;
  if (implementation && typeof implementation.transform !== "function") {
    done(
      new TypeError(
        `esbuild-loader: options.implementation.transform must be an ESBuild transform function. Received ${typeof implementation.transform}`
      )
    );
    return;
  }
  const transform = implementation?.transform ?? esbuild.transform;
  const { resourcePath } = this;
  const transformOptions = {
    ...esbuildTransformOptions,
    target: options.target ?? "es2015",
    loader: options.loader ?? "default",
    sourcemap: this.sourceMap,
    sourcefile: resourcePath
  };
  const isDependency = resourcePath.includes(`${path.sep}node_modules${path.sep}`);
  if (!("tsconfigRaw" in transformOptions) && (!isDependency || tsExtensionsPattern.test(resourcePath))) {
    if (!isDependency && tsconfigPath) {
      const tsconfigFullPath = path.resolve(tsconfigPath);
      const cacheKey = `esbuild-loader:${tsconfigFullPath}`;
      let tsconfig = tsconfigCache.get(cacheKey);
      if (!tsconfig) {
        tsconfig = {
          config: getTsconfig.parseTsconfig(tsconfigFullPath, tsconfigCache),
          path: tsconfigFullPath
        };
        tsconfigCache.set(cacheKey, tsconfig);
      }
      const filesMatcher = getTsconfig.createFilesMatcher(tsconfig);
      const matches = filesMatcher(resourcePath);
      if (!matches) {
        this.emitWarning(
          new Error(`esbuild-loader] The specified tsconfig at "${tsconfigFullPath}" was applied to the file "${resourcePath}" but does not match its "include" patterns`)
        );
      }
      transformOptions.tsconfigRaw = tsconfig.config;
    } else {
      let tsconfig;
      try {
        tsconfig = getTsconfig.getTsconfig(resourcePath, "tsconfig.json", tsconfigCache);
      } catch (error) {
        if (error instanceof Error) {
          const tsconfigError = new Error(`[esbuild-loader] Error parsing tsconfig.json:
${error.message}`);
          if (isDependency) {
            this.emitWarning(tsconfigError);
          } else {
            return done(tsconfigError);
          }
        }
      }
      if (tsconfig) {
        const fileMatcher = getTsconfig.createFilesMatcher(tsconfig);
        transformOptions.tsconfigRaw = fileMatcher(resourcePath);
      }
    }
  }
  transformOptions.supported = {
    "dynamic-import": true,
    ...transformOptions.supported
  };
  try {
    const { code, map } = await transform(source, transformOptions);
    done(null, code, map && JSON.parse(map));
  } catch (error) {
    done(error);
  }
}

var version = "4.3.0";

const isJsFile = /\.[cm]?js(?:\?.*)?$/i;
const isCssFile = /\.css(?:\?.*)?$/i;
const pluginName = "EsbuildPlugin";
const transformAssets = async (options, transform, compilation, useSourceMap) => {
  const { compiler } = compilation;
  const sources = "webpack" in compiler && compiler.webpack.sources;
  const SourceMapSource = sources ? sources.SourceMapSource : webpackSources.SourceMapSource;
  const RawSource = sources ? sources.RawSource : webpackSources.RawSource;
  const {
    css: minifyCss,
    include,
    exclude,
    implementation,
    ...transformOptions
  } = options;
  const minimized = transformOptions.minify || transformOptions.minifyWhitespace || transformOptions.minifyIdentifiers || transformOptions.minifySyntax;
  const assets = compilation.getAssets().filter((asset) => (
    // Filter out already minimized
    !asset.info.minimized && (isJsFile.test(asset.name) || minifyCss && isCssFile.test(asset.name)) && ModuleFilenameHelpers.matchObject(
      {
        include,
        exclude
      },
      asset.name
    )
  ));
  await Promise.all(assets.map(async (asset) => {
    const assetIsCss = isCssFile.test(asset.name);
    let source;
    let map = null;
    if (useSourceMap) {
      if (asset.source.sourceAndMap) {
        const sourceAndMap = asset.source.sourceAndMap();
        source = sourceAndMap.source;
        map = sourceAndMap.map;
      } else {
        source = asset.source.source();
        if (asset.source.map) {
          map = asset.source.map();
        }
      }
    } else {
      source = asset.source.source();
    }
    const sourceAsString = source.toString();
    const result = await transform(sourceAsString, {
      ...transformOptions,
      loader: assetIsCss ? "css" : transformOptions.loader,
      sourcemap: useSourceMap,
      sourcefile: asset.name
    });
    if (result.legalComments) {
      compilation.emitAsset(
        `${asset.name}.LEGAL.txt`,
        new RawSource(result.legalComments)
      );
    }
    compilation.updateAsset(
      asset.name,
      // @ts-expect-error complex webpack union type for source
      result.map ? new SourceMapSource(
        result.code,
        asset.name,
        // @ts-expect-error it accepts strings
        result.map,
        sourceAsString,
        map,
        true
      ) : new RawSource(result.code),
      {
        ...asset.info,
        minimized
      }
    );
  }));
};
class EsbuildPlugin {
  options;
  constructor(options = {}) {
    const { implementation } = options;
    if (implementation && typeof implementation.transform !== "function") {
      throw new TypeError(
        `[${pluginName}] implementation.transform must be an esbuild transform function. Received ${typeof implementation.transform}`
      );
    }
    this.options = options;
  }
  apply(compiler) {
    const {
      implementation,
      ...options
    } = this.options;
    const transform = implementation?.transform ?? esbuild.transform;
    if (!("format" in options)) {
      const { target } = compiler.options;
      const isWebTarget = Array.isArray(target) ? target.includes("web") : target === "web";
      const wontGenerateHelpers = !options.target || (Array.isArray(options.target) ? options.target.length === 1 && options.target[0] === "esnext" : options.target === "esnext");
      if (isWebTarget && !wontGenerateHelpers) {
        options.format = "iife";
      }
    }
    const usedAsMinimizer = compiler.options.optimization?.minimizer?.includes?.(this);
    if (usedAsMinimizer && !("minify" in options || "minifyWhitespace" in options || "minifyIdentifiers" in options || "minifySyntax" in options)) {
      options.minify = compiler.options.optimization?.minimize;
    }
    compiler.hooks.compilation.tap(pluginName, (compilation) => {
      const meta = JSON.stringify({
        name: "esbuild-loader",
        version,
        options
      });
      compilation.hooks.chunkHash.tap(
        pluginName,
        (_, hash) => hash.update(meta)
      );
      let useSourceMap = false;
      compilation.hooks.finishModules.tap(
        pluginName,
        (modules) => {
          const firstModule = Array.isArray(modules) ? modules[0] : modules.values().next().value;
          if (firstModule) {
            useSourceMap = firstModule.useSourceMap;
          }
        }
      );
      if ("processAssets" in compilation.hooks) {
        compilation.hooks.processAssets.tapPromise(
          {
            name: pluginName,
            // @ts-expect-error undefined on Function type
            stage: compilation.constructor.PROCESS_ASSETS_STAGE_OPTIMIZE_SIZE,
            additionalAssets: true
          },
          () => transformAssets(options, transform, compilation, useSourceMap)
        );
        compilation.hooks.statsPrinter.tap(pluginName, (statsPrinter) => {
          statsPrinter.hooks.print.for("asset.info.minimized").tap(
            pluginName,
            (minimized, { green, formatFlag }) => minimized ? green(formatFlag("minimized")) : void 0
          );
        });
      } else {
        compilation.hooks.optimizeChunkAssets.tapPromise(
          pluginName,
          () => transformAssets(options, transform, compilation, useSourceMap)
        );
      }
    });
  }
}

exports.EsbuildPlugin = EsbuildPlugin;
exports.default = ESBuildLoader;
