# Copyright (C) 2022 LuncyBloont

from copyreg import constructor
from random import random
from random import randint
from random import seed as set_seed
from typing import List
import bpy
import bmesh
import math
import traceback
import sys

bl_info = {
    "name": "Ivy Anywhere",
    "author": "LuncyBloont",
    "version": (1, 0, 1),
    "blender": (2, 80, 0),
    "location": "View3D > Add > Mesh > plant Ivy",
    "description": "A Blender addon for Vine-like plants generation.",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}

from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Matrix, Vector, bvhtree

def get_closest(p, btree, rand, obj):
    np = p.to_4d()
    np.w = 1
    mat = Matrix(obj.matrix_world)
    imat = mat.inverted()
    np = imat @ np
    location, normal, *_ = btree.find_nearest(np.xyz)
    if location == None:
        print('Find None')
        return (imat @ mat @ p, Vector((0, 0, 1)))
    
    nmove = Vector((random() - 0.5, random() - 0.5, random() - 0.5)) * 2
    rl = location.to_4d()
    rn = (normal + nmove * rand).normalized().to_4d()
    rl.w = 1
    rn.w = 0
    return ((mat @ rl).xyz, (imat.transposed() @ rn).normalized().xyz)

ivyseed = 0

def rand_XY(size):
    r = random() * math.pi * 2
    l = random() * size
    return (math.cos(r) * l, math.sin(r) * l)

def rand_p(size, curve_x, curve_y, curve_z):
    return Vector((rand11(curve_x) * size, rand11(curve_y) * size, rand11(curve_z) * size))

def move_leaf(v, dis, fall, fall_c):
    v.z -= math.pow(abs(dis), fall_c) * fall

def put_leaf(pos, size, noise, curve_t, mesh, tbn, dis, rotate, fall, fall_c, swing, leaf_fix_rotate):
    
    up = Vector((0, 0, 1))
    foward = Vector((tbn[2].x, tbn[2].y, 0)).normalized()
    right = foward.cross(up).normalized()
    
    foward = foward + right * rand11(1) * tbn[2].z
    foward.normalize()
    right = foward.cross(up).normalized()
    
    offset = rand_p(noise, curve_t[0], curve_t[1], curve_t[2])
    offset.y += noise * curve_t[1] + size / 2
    where = pos + offset.z * up + offset.y * foward + offset.x * right
    
    v0 = (Vector((-1, 1, 0)) * size).to_4d()
    v1 = (Vector((1, 1, 0)) * size).to_4d()
    v2 = (Vector((1, -1, 0)) * size).to_4d()
    v3 = (Vector((-1, -1, 0)) * size).to_4d()
    
    r_on_z = math.atan2(tbn[2].y, tbn[2].x) + leaf_fix_rotate * math.pi / 180
    
    cosrz = math.cos(r_on_z)
    sinrz = math.sin(r_on_z)
    
    dis2 = math.dist((0, 0), (offset.x, offset.y))
    
    cosrx = math.cos(dis2 * rotate)
    sinrx = math.sin(dis2 * rotate)
    
    rz_mat = Matrix([
        [cosrz, -sinrz, 0, 0], 
        [sinrz, cosrz, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rx_mat = Matrix([
        [1, 0, 0, 0],
        [0, cosrx, -sinrx, 0], 
        [0, sinrx, cosrx, 0],
        [0, 0, 0, 1]
    ])
    
    v0.z -= rand11(1) * swing
    v1.z -= rand11(1) * swing
    v2.z -= rand11(1) * swing
    v3.z -= rand11(1) * swing
    
    v0 = rz_mat @ rx_mat @ v0
    v1 = rz_mat @ rx_mat @ v1
    v2 = rz_mat @ rx_mat @ v2
    v3 = rz_mat @ rx_mat @ v3
    
    move_leaf(v0, dis2, fall, fall_c)
    move_leaf(v1, dis2, fall, fall_c)
    move_leaf(v2, dis2, fall, fall_c)
    move_leaf(v3, dis2, fall, fall_c)
    
    p0 = mesh.verts.new(v0.xyz + where)
    p1 = mesh.verts.new(v1.xyz + where)
    p2 = mesh.verts.new(v2.xyz + where)
    p3 = mesh.verts.new(v3.xyz + where)
    
    leaf = mesh.faces.new([p0, p1, p2, p3])
    leaf.normal_update()
    if leaf.normal.z < 0:
        leaf.normal_flip()
    
    return leaf
    
def lerp_2(index, size, mins, maxs):
    lp = index / size
    return mins * (1 - lp) + maxs * lp

_rand11_case = True
def rand11(rc):
    global _rand11_case
    rd = random() * 2 - 1
    if _rand11_case:
        rd = -rd
        _rand11_case = False
    else:
        _rand11_case = True
    r = math.pow(abs(rd), rc) * (-1 if rd <= 0 else 1)
    return r

def normal_func(center, height, scale, x):
    pi2q = math.sqrt(2 * math.pi)
    th = 1 / (height * pi2q)
    return 1 / pi2q / th * math.exp(-math.pow((x - center) * scale, 2) / (2 * th * th))

def add_object(self, context: bpy.types.Context):
    
    # bpy.context.scene.cursor
    global _rand11_case
    _rand11_case = True
    
    curve = bpy.data.curves.new('Ivy0', 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = self.line_size if not self.use_default else 0.005
    
    if bpy.context.active_object is None:
        return
    
    solid = bpy.context.active_object
    dg = bpy.context.evaluated_depsgraph_get()
    postSolid = solid.evaluated_get(dg)
    solid_mesh = postSolid.to_mesh()
    
    bm = bmesh.new()
    bm.from_mesh(solid_mesh)
    
    btree = bvhtree.BVHTree.FromBMesh(bm)
    
    leaf_mesh = bpy.data.meshes.new(name="Leaf")
    bleaf = bmesh.new()
    bleaf.from_mesh(leaf_mesh)
    # pos, size, noise, curve_t, mesh, mat
    
    if solid is None or solid.to_mesh() is None:
        return
    
    cursor_mat = Matrix(bpy.context.scene.cursor.matrix)
    icursor_mat = Matrix.Translation(cursor_mat.to_translation()).inverted()
    
    # properties of the Ivy
    float_bias = self.float_bias if not self.use_default else 0.02
    max_pcount = self.max_pcount if not self.use_default else 52
    min_size = self.min_size if not self.use_default else 0.3
    max_size = self.max_size if not self.use_default else 1.0
    normal_rand = self.normal_rand if not self.use_default else 0.07
    gravity = self.gravity if not self.use_default else 0.02
    round = self.round if not self.use_default else 0.006
    ivy_count = self.ivy_count if not self.use_default else 6
    grow_force = self.grow_force if not self.use_default else 0.02
    grow_speed = self.grow_speed if not self.use_default else 0.06
    hand = self.hand if not self.use_default else 0.2
    close_force = self.close_force if not self.use_default else 0.02
    noise_on_right = self.noise_on_right if not self.use_default else 0.03
    noise_on_normal = self.noise_on_normal if not self.use_default else 0.04
    noise_speed = self.noise_speed if not self.use_default else 0.1
    hand_curve = self.hand_curve if not self.use_default else 0.4
    close_curve = self.close_curve if not self.use_default else 0.6
    close_force_max_position = self.close_force_max_position if not self.use_default else 0.4
    wall_fac = self.wall_fac if not self.use_default else 2.6
    rand_change_on_right = self.rand_change_on_right if not self.use_default else 1.0
    rand_change_on_normal = self.rand_change_on_normal if not self.use_default else 1.0
    leaf_count = self.leaf_count if not self.use_default else 20
    leaf_size = self.leaf_size if not self.use_default else 0.03
    leaf_radius = self.leaf_radius if not self.use_default else 0.06
    leaf_thickness_c = self.leaf_thickness_c if not self.use_default else 30
    leaf_rotate = self.leaf_rotate if not self.use_default else 2
    leaf_fall = self.leaf_fall if not self.use_default else 0.3
    leaf_fall_curve = self.leaf_fall_curve if not self.use_default else 2
    leaf_swing = self.leaf_swing if not self.use_default else 0.02
    leaf_fix_rotate = self.leaf_fix_rotate if not self.use_default else 90
    leaf_count_curve = self.leaf_count_curve if not self.use_default else 'y = m.pow(x, 1.5)'
    
    leaf_bag = []
    
    leaf_put_args = []
    
    global ivyseed
    print('+', ivyseed, '+')
    pseed = ivyseed
    set_seed(pseed)
    seedlist = []
    for ivy in range(ivy_count):
        seedlist.append(randint(0, 999999))
    
    for ivy in range(ivy_count):
        set_seed(pseed + seedlist[ivy])
        print('pseed', pseed, pseed + seedlist[ivy])
        sps = curve.splines.new('BEZIER')
        
        climb = bpy.context.scene.cursor.location
        old = climb.copy()
        
        grow_to = Vector((0, 0, grow_speed))
        
        G = Vector((0, round, -gravity))
        
        for i in range(0, max_pcount, 1):
            
            if i > 0:
                sps.bezier_points.add(1)
            
            clst, normal = get_closest(climb, btree, normal_rand, solid)
            
            ''' # test space trans START
            g = clst.to_4d()
            
            g = icursor_mat @ g
            
            gn = normal.to_4d()
            gn.w = 0
            
            gn = icursor_mat @ gn
            
            sps.bezier_points[i].co = g.xyz
            sps.bezier_points[i].handle_left = g.xyz
            sps.bezier_points[i].handle_right = g.xyz
            sps.bezier_points.add(1)
            sps.bezier_points[i + 1].co = g.xyz + gn.xyz
            sps.bezier_points[i + 1].handle_left = g.xyz + gn.xyz
            sps.bezier_points[i + 1].handle_right = g.xyz + gn.xyz
            
            climb = climb + Vector((0, 0, 0.02))
            
            continue
            ''' # test space trans END
            
            tangent = normal.cross(grow_to).normalized()
            biotangent = tangent.cross(normal)
            
            dist_to_surface = (climb - clst).dot(normal) - float_bias
            
            hit_wall = False
            if dist_to_surface < 0:
                climb += -dist_to_surface * normal
                hit_wall = True
                dist_to_surface = 0
            
            g = Vector(climb).to_4d()
            o = Vector(old).to_4d()
            
            g = icursor_mat @ g
            o = icursor_mat @ o
            
            sps.bezier_points[i].co = g.xyz
            
            lerp = i / max_pcount
            sps.bezier_points[i].radius = 1 - lerp * max_size + lerp * min_size
            
            if i == 0:
                sps.bezier_points[i].handle_left = g.xyz
                sps.bezier_points[i].handle_right = g.xyz
            else:
                sps.bezier_points[i].handle_left = (g.xyz + o.xyz) / 2
                sps.bezier_points[i].handle_right = g.xyz - (g.xyz + o.xyz) / 2 + g.xyz

            # calculate next position:
            old = climb.copy()
            
            force = Vector((0, 0, 0)) + tangent * rand11(rand_change_on_right) * noise_on_right + \
                normal * noise_on_normal * rand11(rand_change_on_normal)
            if hit_wall:
                # hit the wall
                force += Vector((0, 0, 1)) * grow_force
            else:
                # not hit wall
                force += Vector((0, 0, 1)) * grow_force / (1 + dist_to_surface * wall_fac)
                force += -normal * (1 + -normal.z) * \
                    normal_func(hand, 1, hand_curve, dist_to_surface) * \
                    normal_func(close_force_max_position, close_force, close_curve, i / max_pcount)
                grow_to += G
            
            grow_to += force + tangent * rand11(rand_change_on_right) * noise_on_right * noise_speed + \
                normal * noise_on_normal * rand11(rand_change_on_normal) * noise_speed
            
            if grow_to.length > grow_speed:
                grow_to = grow_to.normalized() * grow_speed
            
            climb = climb + grow_to
            
            # put leaf
            leaf_good = (normal.z + 1) / 2
            code_tmp0 = {'y': 1, 'x': i / max_pcount, 'm': math}
            exec(leaf_count_curve, code_tmp0)
            leaf_c = int(leaf_good * leaf_count * random() * code_tmp0['y'])
            for leafi in range(leaf_c):
                # aleaf = put_leaf(g.xyz, leaf_size, leaf_radius * leaf_good * code_tmp0['y'], \
                #     (1, 1, leaf_thickness), bleaf, (tangent, biotangent, normal), dist_to_surface, \
                #     leaf_rotate, leaf_fall, leaf_fall_curve, leaf_swing, leaf_fix_rotate)
                leaf_put_args.append((g.xyz, leaf_size, leaf_radius * leaf_good * code_tmp0['y'], \
                    (1, 1, leaf_thickness_c), bleaf, (tangent, biotangent, normal), dist_to_surface, \
                    leaf_rotate, leaf_fall, leaf_fall_curve, leaf_swing, leaf_fix_rotate))
                # leaf_bag.append(aleaf)
    
    set_seed(ivyseed + 581234)
    
    for args in leaf_put_args:
        aleaf = put_leaf(*args)
        leaf_bag.append(aleaf)
    
    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    
    bleaf.to_mesh(leaf_mesh)
    
    uvs = [Vector((0, 1)), Vector((1, 1)), Vector((1, 0)), Vector((0, 0))]
    
    uv_layer = leaf_mesh. uv_layers.new().data
    print(type(uv_layer))
    for li in leaf_bag:
        for i in range(len(li.loops)):
            ix = li.loops[i].index
            uv_layer[ix].uv = uvs[i]
    
    ivy = bpy.context.blend_data.objects.new("Ivy", curve)
    ivy.location = bpy.context.scene.cursor.location
    bpy.context.scene.collection.objects.link(ivy)
    
    
    ivy_leaf = bpy.context.blend_data.objects.new("Ivy_leaf", leaf_mesh)
    ivy_leaf.location = bpy.context.scene.cursor.location
    bpy.context.scene.collection.objects.link(ivy_leaf)

    set_seed(ivyseed)


class OBJECT_OT_add_object(Operator, AddObjectHelper):
    bl_idname = "mesh.add_object"
    bl_label = "plant Ivy"
    bl_options = {'REGISTER', 'UNDO'}

    inseed: bpy.props.IntProperty(name="seed", default=0)
    use_default: bpy.props.BoolProperty(name="default", default=False)
    line_size: bpy.props.FloatProperty(name="line size", default=0.005)
    float_bias: bpy.props.FloatProperty(name="float bias", default=0.02)
    max_pcount: bpy.props.IntProperty(name="max point count", default=52)
    min_size: bpy.props.FloatProperty(name="min size", default=0.3)
    max_size: bpy.props.FloatProperty(name="max size", default=1.0)
    normal_rand: bpy.props.FloatProperty(name="random normal", default=0.07)
    gravity: bpy.props.FloatProperty(name="gravity", default=0.02)
    round: bpy.props.FloatProperty(name="round wall", default=0.006)
    ivy_count: bpy.props.IntProperty(name="count", default=6)
    grow_force: bpy.props.FloatProperty(name="grow force", default=0.02)
    grow_speed: bpy.props.FloatProperty(name="grow speed", default=0.06)
    hand: bpy.props.FloatProperty(name="hand radius", default=0.2)
    close_force: bpy.props.FloatProperty(name="close wall force", default=0.02)
    noise_on_right: bpy.props.FloatProperty(name="noise on right", default=0.03)
    noise_on_normal: bpy.props.FloatProperty(name="noise on normal", default=0.04)
    noise_speed: bpy.props.FloatProperty(name="noise speed", default=0.1)
    hand_curve: bpy.props.FloatProperty(name="hand curve", default=0.4)
    close_curve: bpy.props.FloatProperty(name="near curve", default=0.6)
    close_force_max_position: bpy.props.FloatProperty(name="near force", default=0.4)
    wall_fac: bpy.props.FloatProperty(name="force from wall", default=2.6)
    rand_change_on_right: bpy.props.FloatProperty(name="change on right", default=1.0)
    rand_change_on_normal: bpy.props.FloatProperty(name="change on normal", default=1.0)
    leaf_count: bpy.props.IntProperty(name="leaf count", default=20)
    leaf_size: bpy.props.FloatProperty(name="leaf size", default=0.03)
    leaf_radius: bpy.props.FloatProperty(name="leaf radius", default=0.06)
    leaf_thickness_c: bpy.props.FloatProperty(name="leaf thickness curve", default=30)
    leaf_rotate: bpy.props.FloatProperty(name="leaf rotate", default=2)
    leaf_fall: bpy.props.FloatProperty(name="leaf fall", default=0.3)
    leaf_fall_curve: bpy.props.FloatProperty(name="leaf fall curve",default=2)
    leaf_swing: bpy.props.FloatProperty(name="leaf swing", default=0.02)
    leaf_fix_rotate: bpy.props.FloatProperty(name="leaf rotate fix on z", default=90)
    leaf_count_curve: bpy.props.StringProperty(name="leaf count curve", default='y = m.pow(x, 1.5)')

    def execute(self, context):
        global ivyseed
        try:
            ivyseed = abs(self.inseed) + 304661
            add_object(self, context)

            return {'FINISHED'}
        except Exception as e:
            traceback.print_tb(sys.exc_info()[2])


def add_object_button(self, context):
    self.layout.operator(
        OBJECT_OT_add_object.bl_idname,
        text="plant Ivy",
        icon='PLUGIN')


def add_object_manual_map():
    url_manual_prefix = "https://github.com/LuncyBloont/IvyAnywhere"
    url_manual_mapping = (
        ("bpy.ops.mesh.add_object", "https://github.com/LuncyBloont/IvyAnywhere"),
    )
    return url_manual_prefix, url_manual_mapping

def register():
    bpy.utils.register_class(OBJECT_OT_add_object)
    bpy.utils.register_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.append(add_object_button)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_add_object)
    bpy.utils.unregister_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)


if __name__ == "__main__":
    register()
