#!/usr/bin/env node
/** Build a private result deck by editing named objects in a supplied PPTX.
 * Install no packages here: use the bundled runtime via RUNTIME_NODE_MODULES.
 * Keep the spec, template, scientific figures, and generated files outside Git.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const specPath = process.argv[2];
if (!specPath || !process.env.RUNTIME_NODE_MODULES) {
  throw new Error('Usage: RUNTIME_NODE_MODULES=... node scripts/build_template_presentation.mjs private-spec.json [--finalize]');
}
const requireRuntime = createRequire(path.join(process.env.RUNTIME_NODE_MODULES, 'package.json'));
const {FileBlob, Presentation, PresentationFile} = requireRuntime('@oai/artifact-tool');
const sharp = requireRuntime('sharp');
const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
const root = path.resolve(path.dirname(specPath));
const locate = value => path.resolve(root, value);
const build = locate(spec.buildDirectory);
const template = locate(spec.template);
const candidate = path.join(build, 'candidate.pptx');
await fs.mkdir(build, {recursive: true});

if (process.argv.includes('--finalize')) {
  const skill = process.env.PRESENTATION_SKILL_DIR;
  if (!skill || !process.env.RUNTIME_PYTHON) throw new Error('Finalization requires PRESENTATION_SKILL_DIR and RUNTIME_PYTHON');
  const {finalizePresentation} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
  const finalPath = locate(spec.finalPath);
  await fs.mkdir(path.dirname(finalPath), {recursive: true});
  const result = await finalizePresentation({
    workspaceDir: locate(spec.workspaceDirectory), candidatePath: candidate, finalPath,
    pythonExecutable: process.env.RUNTIME_PYTHON,
    integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
    explicitTotalSlideCount: spec.slides.length,
    requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [],
    sourceTemplatePath: template,
    layoutArgs: ['--expected-slide-size-emu', spec.slideSizeEmu, '--validate-heading-fit'],
    fontPolicy: {basis:'reference', families:[spec.fontFamily], referencePath:template,
      referenceSha256:crypto.createHash('sha256').update(await fs.readFile(template)).digest('hex')},
    verifyArtifactToolImport: true,
    receiptPath: path.join(build, 'validation.json'),
  });
  console.log(JSON.stringify(result, null, 2));
  const deck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
  for (const [i,slide] of deck.slides.items.entries()) {
    const png = await deck.export({slide,format:'png',scale:1.5});
    await fs.writeFile(path.join(build,`final-slide-${i+1}.png`), new Uint8Array(await png.arrayBuffer()));
  }
  process.exit(0);
}

const source = await PresentationFile.importPptx(await FileBlob.load(template));
const proto = source.toProto();
const sourceSlides = proto.slides;
proto.slides = spec.slides.map((entry,i) => {
  if (!sourceSlides[entry.templateSlide-1]) throw new Error(`Unknown source slide ${entry.templateSlide}`);
  const slide = structuredClone(sourceSlides[entry.templateSlide-1]);
  slide.id = crypto.randomUUID().replaceAll('-',''); slide.index = i;
  slide.elements = slide.elements.filter(element=>!(entry.removeNames??[]).includes(element.name));
  // IDs in imported templates may be scoped to a slide. Make cloned IDs unique.
  const assignIds = elements => elements.forEach(element => {
    element.id = `${i+1}_${element.id}`;
    if (element.children) assignIds(element.children);
  });
  assignIds(slide.elements);
  return slide;
});
const presentation = Presentation.load(proto);
for (const [i,entry] of spec.slides.entries()) {
  const slide = presentation.slides.items[i];
  const named = name => {
    const matches = slide.shapes.items.filter(s=>s.name===name);
    if (matches.length!==1) throw new Error(`Slide ${i+1}: expected one shape named ${name}, found ${matches.length}`);
    return matches[0];
  };
  for (const edit of entry.edits??[]) {
    const shape = named(edit.name);
    if (edit.position) shape.position = {...shape.position,...edit.position};
    if (edit.text!==undefined) shape.text = edit.text;
    if (edit.style) shape.text.style = {typeface:spec.fontFamily,...edit.style};
    if (edit.text!==undefined) shape.text.insets=0;
  }
  for (const item of entry.texts??[]) {
    const shape=slide.shapes.add({name:item.name,geometry:'textbox',position:item.position,
      fill:'none',line:{fill:'none',width:0}});
    shape.text=item.text;
    shape.text.style={typeface:spec.fontFamily,fontSize:24,color:'#001B52',insets:0,
      verticalAlignment:'middle',...item.style};
  }
  for (const item of entry.images??[]) {
    let pipeline=sharp(locate(item.path));
    if (item.crop) pipeline=pipeline.extract(item.crop);
    const bytes=await pipeline.png().toBuffer();
    slide.images.add({name:item.name,blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),
      contentType:'image/png',alt:item.alt,fit:'contain',position:item.position});
  }
  slide.speakerNotes.text=entry.notes??'';
}
await (await PresentationFile.exportPptx(presentation)).save(candidate);
await fs.writeFile(path.join(build,'deck-inspection.ndjson'),
  (await presentation.inspect({kind:'slide,textbox,image,notes',maxChars:100000})).ndjson);
for (const [i,slide] of presentation.slides.items.entries()) {
  const png=await presentation.export({slide,format:'png',scale:1.5});
  await fs.writeFile(path.join(build,`draft-slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({candidate,slideCount:presentation.slides.items.length}));
