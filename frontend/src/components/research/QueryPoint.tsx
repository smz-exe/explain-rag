"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { hexToThreeColor, visualization } from "@/lib/design-tokens";

interface QueryPointProps {
  coords: [number, number, number];
}

/**
 * Query point marker - octahedron shape to distinguish from paper spheres.
 * Includes pulsing animation and label.
 */
export function QueryPoint({ coords }: QueryPointProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);

  // Rotate and pulse animation
  useFrame((state) => {
    if (!meshRef.current || !materialRef.current) return;

    // Slow rotation
    meshRef.current.rotation.y += 0.01;

    // Pulse effect
    const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 1;
    meshRef.current.scale.setScalar(pulse);

    // Emissive pulse
    materialRef.current.emissiveIntensity = 0.3 + Math.sin(state.clock.elapsedTime * 3) * 0.2;
  });

  const color = hexToThreeColor(visualization.queryPoint.color);

  return (
    <group position={coords}>
      <mesh ref={meshRef}>
        <octahedronGeometry args={[visualization.queryPoint.size, 0]} />
        <meshStandardMaterial
          ref={materialRef}
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          roughness={0.3}
          metalness={0.2}
        />
      </mesh>

      {/* Label */}
      <Html
        position={[0, visualization.queryPoint.size + 0.15, 0]}
        center
        style={{ pointerEvents: "none" }}
      >
        <div className="whitespace-nowrap rounded bg-black px-2 py-1 text-xs font-medium text-white shadow-lg">
          Query
        </div>
      </Html>
    </group>
  );
}
